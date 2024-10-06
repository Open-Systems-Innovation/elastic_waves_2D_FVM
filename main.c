static char help[] =
    "2D FVM acoustic wave problem with custom Riemann Solver and Slope Limiter\n";
/*F
We use a second order Godunov finite volume method (FVM) to simulate 2D acoustic
wave propagation. Our simple upwinded residual evaluation loops over all mesh
faces and uses a Riemann solver to produce the flux given the face geometry and
cell values,

\begin{equation}
  f_i = \mathrm{riemann}(\mathrm{phys}, p_\mathrm{centroid}, \hat n, x^L, x^R)
\end{equation}

and then update the cell values given the cell volume.
\begin{eqnarray}
    f^L_i &-=& \frac{f_i}{vol^L} \\
    f^R_i &+=& \frac{f_i}{vol^R}
\end{eqnarray}

In this code we solve the elastic wave equations in 2D to simulate pressure
waves in a solid.
\begin{equation}
\frac{\partial}{\partial t}
\begin{bmatrix}
p_x \\
p_y \\
\sigma_{xx} \\
\sigma_{yy} \\
\sigma_{xy}
\end{bmatrix}
+
\frac{\partial}{\partial x}
\begin{bmatrix}
\sigma_{xx} \\
\sigma_{xy} \\
2\mu \frac{\partial v_x}{\partial x} + \lambda (\nabla \cdot \vec{v}) \\
0 \\
\mu \frac{\partial v_y}{\partial x}
\end{bmatrix}
+
\frac{\partial}{\partial y}
\begin{bmatrix}
\sigma_{xy} \\
\sigma_{yy} \\
0 \\
2\mu \frac{\partial v_y}{\partial y} + \lambda (\nabla \cdot \vec{v}) \\
\mu \frac{\partial v_x}{\partial y}
\end{bmatrix}
= 0
\end{equation}

A representative Riemann solver for the elastic wave equations is given in the riemann() function,
\begin{eqnarray}
  f^{L,R}_h    &=& uh^{L,R} \cdot \hat n \\
  f^{L,R}_{uh} &=& \frac{f^{L,R}_h}{h^{L,R}} uh^{L,R} + g (h^{L,R})^2 \hat n \\
  c^{L,R}      &=& \sqrt{g h^{L,R}} \\
  s            &=& \max\left( \left|\frac{uh^L \cdot \hat n}{h^L}\right| + c^L, \left|\frac{uh^R \cdot \hat n}{h^R}\right| + c^R \right) \\
  f_i          &=& \frac{A_\mathrm{face}}{2} \left( f^L_i + f^R_i + s \left( x^L_i - x^R_i \right) \right)
\end{eqnarray}
where $c$ is the local gravity wave speed and $f_i$ is a Rusanov flux.

The mesh is read in from an Gmsh file.

The example also shows how to handle AMR in a time-dependent TS solver.
F*/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Include statements 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>
#include <petscoptions.h>
#include <petsc/private/dmpleximpl.h> /* For norm and dot */

/* import custom elastic wave rieman solver*/
#include "elastic_physics.h"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Model Structure 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* model structure includes boundary conditions, initial conditions, and functionals of interest. It is
 * discretization-independent, but its members depend on the scenario being solved. */
typedef struct _n_Model *Model;

struct _n_Model {
  MPI_Comm         comm; /* Does not do collective communication, but some error conditions can be collective */
  Physics          physics;
  PetscInt         maxComputed;
  PetscInt         numMonitored;
  PetscInt         numCall;
  PetscReal        maxspeed; /* estimate of global maximum speed (for CFL calculation) */
  PetscErrorCode (*errorIndicator)(PetscInt, PetscReal, PetscInt, const PetscScalar[], const PetscScalar[], PetscReal *, void *);
  void *errorCtx;
};

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   User Context 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
typedef struct _n_User *User;

struct _n_User {
  char     infile[PETSC_MAX_PATH_LEN];  /* Input mesh filename */
  char     outfile[PETSC_MAX_PATH_LEN]; /* Dump/reload mesh filename */
  Model     model;
  PetscInt  monitorStepOffset;
  PetscInt  vtkInterval;                        /* For monitor */
  PetscBool vtkmon;
};

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Boundary Conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
PetscErrorCode ZeroBoundaryCondition(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  /* dim - the spatial dimension
     time - current time
     x[] - coordinates of the current point
     Nc - number of constant parameters 
     u[] - each field evaluated at the current point
     ctx - extra user context
   */
  const PetscInt Ncomp = dim;
  PetscInt       comp;  // iterates over components of the field

  for (comp = 0; comp < Ncomp; ++comp) {
    u[comp] = 0;
  }

  return PETSC_SUCCESS;
}

static PetscErrorCode SetUpBC(DM dm, PetscDS ds, Physics phys)
{
  DMLabel        label;
  PetscInt       field = 0;   // we're working with a single field
  
  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "Face Sets", &label));
  // Add Dirichlet boundary condition (Essential) on Left (Label = 4)
  PetscInt left_values[] = {4};  // Physical group label for "Left"
  PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "Left Boundary", label, 1, left_values,
                     field, 0, NULL, (void (*)(void))ZeroBoundaryCondition, NULL, NULL, NULL);

  // Add Dirichlet boundary condition (Essential) on Right (Label = 2)
  PetscInt right_values[] = {2};  // Physical group label for "Right"
  PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "Right Boundary", label, 1, right_values,
                            field, 0, NULL, (void (*)(void))ZeroBoundaryCondition, NULL, NULL, NULL);

  // Add Dirichlet boundary condition (Essential) on Bottom (Label = 1)
  PetscInt bottom_values[] = {1};  // Physical group label for "Bottom"
  PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "Bottom Boundary", label, 1, bottom_values,
                            field, 0, NULL, (void (*)(void))ZeroBoundaryCondition, NULL, NULL, NULL);

  // Add Dirichlet boundary condition (Essential) on Top (Label = 3)
  PetscInt top_values[] = {3};  // Physical group label for "Top"
  PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "Top Boundary", label, 1, top_values,
                            field, 0, NULL, (void (*)(void))ZeroBoundaryCondition, NULL, NULL, NULL);

  DMViewFromOptions(dm, NULL, "-after_ds");
  PetscDSViewFromOptions(ds, NULL, "-ds_view");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Initial Conditions 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
PetscErrorCode zero_vector(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  /*
    dim - The spatial dimension
    time - The time at which to sample
    x - The coordinates
    Nc - The number of components
    u - The output field values
    ctx - optional user-defined function context
   */
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 0.0;
  return 0;
}

PetscErrorCode SetInitialConditions(DM dm, Vec X, User user)
{

  PetscFunctionBeginUser;
  PetscErrorCode (*funcs[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {zero_vector};
  PetscCall(DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Output to VTK to view in Paraview 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
static PetscErrorCode OutputVTK(DM dm, const char *filename, PetscViewer *viewer)
{
  PetscFunctionBeginUser;
  PetscCall(PetscViewerCreate(PetscObjectComm((PetscObject)dm), viewer));
  PetscCall(PetscViewerSetType(*viewer, PETSCVIEWERVTK));
  PetscCall(PetscViewerFileSetName(*viewer, filename));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorVTK(TS ts, PetscInt stepnum, PetscReal time, Vec X, void *ctx)
{
  User        user = (User)ctx;
  DM          dm, plex;
  PetscViewer viewer;
  char        filename[PETSC_MAX_PATH_LEN], *ftable = NULL;
  PetscReal   xnorm;
  PetscBool   rollback;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepRollBack(ts, &rollback));
  if (rollback) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectSetName((PetscObject)X, "u"));
  PetscCall(VecGetDM(X, &dm));
  PetscCall(VecNorm(X, NORM_INFINITY, &xnorm));

  if (stepnum >= 0) stepnum += user->monitorStepOffset;
  if (stepnum >= 0) { /* No summary for final time */
    Model              mod = user->model;
    Vec                cellgeom;
    PetscInt           c, cStart, cEnd, fcount, i;
    size_t             ftableused, ftablealloc;
    const PetscScalar *cgeom, *x;
    DM                 dmCell;
    DMLabel            vtkLabel;
    PetscReal         *fmin, *fmax, *fintegral, *ftmp;

    PetscCall(DMConvert(dm, DMPLEX, &plex));
    PetscCall(DMPlexGetGeometryFVM(plex, NULL, &cellgeom, NULL));
    fcount = mod->maxComputed + 1;
    PetscCall(PetscMalloc4(fcount, &fmin, fcount, &fmax, fcount, &fintegral, fcount, &ftmp));
    for (i = 0; i < fcount; i++) {
      fmin[i]      = PETSC_MAX_REAL;
      fmax[i]      = PETSC_MIN_REAL;
      fintegral[i] = 0;
    }
    PetscCall(VecGetDM(cellgeom, &dmCell));
    PetscCall(DMPlexGetSimplexOrBoxCells(dmCell, 0, &cStart, &cEnd));
    PetscCall(VecGetArrayRead(cellgeom, &cgeom));
    PetscCall(VecGetArrayRead(X, &x));
    PetscCall(DMGetLabel(dm, "vtk", &vtkLabel));
    for (c = cStart; c < cEnd; ++c) {
      PetscFVCellGeom   *cg;
      const PetscScalar *cx     = NULL;
      PetscInt           vtkVal = 0;

      /* not that these two routines as currently implemented work for any dm with a
       * localSection/globalSection */
      PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
      PetscCall(DMPlexPointGlobalRead(dm, c, x, &cx));
      if (vtkLabel) PetscCall(DMLabelGetValue(vtkLabel, c, &vtkVal));
      if (!vtkVal || !cx) continue; /* ghost, or not a global cell */
      for (i = 0; i < mod->numCall; i++) {
      }
      for (i = 0; i < fcount; i++) {
        fmin[i] = PetscMin(fmin[i], ftmp[i]);
        fmax[i] = PetscMax(fmax[i], ftmp[i]);
        fintegral[i] += cg->volume * ftmp[i];
      }
    }
    PetscCall(VecRestoreArrayRead(cellgeom, &cgeom));
    PetscCall(VecRestoreArrayRead(X, &x));
    PetscCall(DMDestroy(&plex));
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, fmin, fcount, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)ts)));
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, fmax, fcount, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)ts)));
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, fintegral, fcount, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)ts)));

    ftablealloc = fcount * 100;
    ftableused  = 0;
    PetscCall(PetscMalloc1(ftablealloc, &ftable));
    for (i = 0; i < mod->numMonitored; i++) {
      size_t         countused;
      char           buffer[256], *p;
      if (i % 3) {
        PetscCall(PetscArraycpy(buffer, "  ", 2));
        p = buffer + 2;
      } else if (i) {
        char newline[] = "\n";
        PetscCall(PetscMemcpy(buffer, newline, sizeof(newline) - 1));
        p = buffer + sizeof(newline) - 1;
      } else {
        p = buffer;
      }
      countused--;
      countused += p - buffer;
      if (countused > ftablealloc - ftableused - 1) { /* reallocate */
        char *ftablenew;
        ftablealloc = 2 * ftablealloc + countused;
        PetscCall(PetscMalloc(ftablealloc, &ftablenew));
        PetscCall(PetscArraycpy(ftablenew, ftable, ftableused));
        PetscCall(PetscFree(ftable));
        ftable = ftablenew;
      }
      PetscCall(PetscArraycpy(ftable + ftableused, buffer, countused));
      ftableused += countused;
      ftable[ftableused] = 0;
    }
    PetscCall(PetscFree4(fmin, fmax, fintegral, ftmp));

    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "% 3" PetscInt_FMT "  time %8.4g  |x| %8.4g  %s\n", stepnum, (double)time, (double)xnorm, ftable ? ftable : ""));
    PetscCall(PetscFree(ftable));
  }
  if (user->vtkInterval < 1) PetscFunctionReturn(PETSC_SUCCESS);
  if ((stepnum == -1) ^ (stepnum % user->vtkInterval == 0)) {
    if (stepnum == -1) { /* Final time is not multiple of normal time interval, write it anyway */
      PetscCall(TSGetStepNumber(ts, &stepnum));
    }
    PetscCall(OutputVTK(dm, filename, &viewer));
    PetscCall(VecView(X, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Initialize Time stepping (TS) object
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
static PetscErrorCode InitializeTS(DM dm, User user, TS *ts)
{
  PetscFunctionBeginUser;
  PetscCall(TSCreate(PetscObjectComm((PetscObject)dm), ts));
  PetscCall(TSSetType(*ts, TSSSP)); // use Runge-Kutta, -ts_ssp_type {rks2,rks3,rk104}
  PetscCall(TSSetDM(*ts, dm));
  if (user->vtkmon) PetscCall(TSMonitorSet(*ts, MonitorVTK, user, NULL));
  PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, user));
  PetscCall(DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, user));
  PetscCall(TSSetMaxTime(*ts, 2.0));
  PetscCall(TSSetExactFinalTime(*ts, TS_EXACTFINALTIME_STEPOVER));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Process User Options 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
static PetscErrorCode ProcessOptions(MPI_Comm comm, User user)
{
  PetscBool flg;

  PetscFunctionBeginUser;
  
  PetscOptionsBegin(comm, "", "My Project's Options", "");
  // get the input file name
  PetscCall(PetscOptionsString("-infile", "The input mesh file", "",
                               user->infile, user->infile,
                               sizeof(user->infile), &flg));
  // get the output file name
  PetscCall(PetscOptionsString("-outfile", "The output mesh file", "",
                               user->outfile, user->outfile,
                               sizeof(user->outfile), &flg));
  // End function
  PetscOptionsEnd();

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Main program
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
int main(int argc, char **argv) {

  // Declare variables
  MPI_Comm          comm;
  PetscDS           ds;
  PetscFV           fv;
  User              user;
  Model             mod;
  Physics           phys;
  DM                dm;
  PetscReal         ftime, cfl, dt, minRadius;
  PetscInt          nsteps;
  PetscInt          dim = 2;
  PetscInt          numComponents = 5;
  TS                ts;
  TSConvergedReason reason;
  Vec               X;

  // Initialize PETSc code
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  comm = PETSC_COMM_WORLD;
      
  PetscCall(PetscNew(&user));
  PetscCall(PetscNew(&user->model));
  PetscCall(PetscNew(&user->model->physics));
  mod           = user->model;
  phys          = mod->physics;
  mod->comm     = comm;

  // Process user options
  PetscCall(ProcessOptions(comm, user));
  
  // Read in 3D mesh from file
  //PetscCall(DMPlexCreateFromFile(comm, user->infile, NULL, PETSC_TRUE,&dm));
  PetscCall(DMPlexCreateGmshFromFile(comm, user->infile, PETSC_TRUE, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-orig_dm_view"));
  PetscCall(DMGetDimension(dm, &dim));
  // create label for boundary conditions
  PetscCall(DMCreateLabel(dm, "Face Sets"));

  // Create finite volume
  PetscCall(PetscFVCreate(comm, &fv));
  PetscCall(PetscFVSetFromOptions(fv));
  PetscCall(PetscFVSetNumComponents(fv, numComponents));
  PetscCall(PetscFVSetSpatialDimension(fv, dim));
  PetscCall(PetscObjectSetName((PetscObject)fv, ""));
  
  // Define component names for pressure and velocity
  {
    // Set names for the components of the field 
    PetscCall(PetscFVSetComponentName(fv, 0, "Stress_11")); // 12 component of stress 
    PetscCall(PetscFVSetComponentName(fv, 1, "Stress_22")); // 22 component of stress 
    PetscCall(PetscFVSetComponentName(fv, 2, "Stress_12")); // 12 component of stress 
    PetscCall(PetscFVSetComponentName(fv, 3, "Velocity_x")); // X component of velocity 
    PetscCall(PetscFVSetComponentName(fv, 4, "Velocity_y")); // Y component of velocity
  }
  PetscCall(PetscFVViewFromOptions(fv, NULL, "-fv_view"));
  
  // add FV to DM
  PetscCall(DMAddField(dm, NULL, (PetscObject)fv));

  // Create the Discrete Systems (DS)
  // each DS object has a set of fields with a PetscVW discretization
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));

  // set the pointwise functions. The actual equations to be enforced
  // on each region
  PetscCall(PetscDSSetRiemannSolver(ds, 0, Elastic_Riemann_Godunov));
  PetscCall(SetUpBC(dm, ds, phys));
  PetscCall(PetscDSSetFromOptions(ds));

  // initialize TS object
  PetscCall(InitializeTS(dm, user, &ts));

  // create solution vector
  PetscCall(DMCreateGlobalVector(dm, &X));
  PetscCall(PetscObjectSetName((PetscObject)X, "solution"));
  PetscCall(SetInitialConditions(dm, X, user));
  
  PetscCall(DMPlexGetGeometryFVM(dm, NULL, NULL, &minRadius));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMDestroy(&dm));
  //PetscCall(MPIU_Allreduce(&phys->maxspeed, &mod->maxspeed, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)ts)));
  
  //dt = cfl * minRadius / mod->maxspeed;
  dt = 0.01;
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSetSolution(ts, X));
  PetscCall(VecDestroy(&X));
  PetscCall(TSViewFromOptions(ts, NULL, "-ts_view"));
  PetscCall(TSSolve(ts, NULL));

  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts, &nsteps));
  PetscCall(TSGetConvergedReason(ts, &reason));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s at time %g after %" PetscInt_FMT " steps\n", TSConvergedReasons[reason], (double)ftime, nsteps));

  // Free objects from memory
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFVDestroy(&fv));
  
  // End main program
  PetscFinalize();
  return 0;
}

