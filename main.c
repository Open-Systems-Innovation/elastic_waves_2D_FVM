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

static void Elastic_Riemann_Godunov(
    PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n,
    const PetscScalar *uL, const PetscScalar *uR, PetscInt numConstants,
    const PetscScalar constants[], PetscScalar *flux, void *ctx)
{
  /* Input Parameters:
       dim          - The spatial dimension
       Nf           - The number of fields
       x            - The coordinates at a point on the interface
       n            - The normal vector to the interface
       uL           - The state vector to the left of the interface
       uR           - The state vector to the right of the interface
       numConstants - number of constant parameters
       constants    - constant parameters
       ctx          - optional user context
       flux         - output array of flux through the interface
  */

  ElasticityContext *material = (ElasticityContext*) ctx; // Cast user context

  PetscReal rho = 1000; // value of water
    PetscReal lambda = 100000;
    PetscReal mu = 500000;
    PetscReal cp = .01;
    PetscReal cs = 0.005;

    PetscReal dsig11, dsig22, dsig12, du, dv;
    PetscReal a1, a2, a3, a4;
    PetscReal det, bulkl, bulkr;

    PetscReal s[4], wave[4][5]; // Assuming 4 waves, 5 components

    // Compute jumps in the states (uL is left, uR is right)
    dsig11 = uL[0] - uR[0]; // sigma11
    dsig22 = uL[1] - uR[1]; // sigma22
    dsig12 = uL[2] - uR[2]; // sigma12
    du = uL[3] - uR[3];     // u (velocity component)
    dv = uL[4] - uR[4];     // v (velocity component)

    // Bulk moduli for left and right states
    bulkl = lambda + 2.0 * mu;
    bulkr = lambda + 2.0 * mu;

    // P-wave strengths
    det = bulkl * cp + bulkr * cp;
    if (det == 0.0) {
        PetscPrintf(PETSC_COMM_SELF, "det=0 in RiemannSolver\n");
    }
    a1 = (cp * dsig11 + bulkr * du) / det;
    a2 = (cp * dsig11 - bulkl * du) / det;

    // S-wave strengths
    det = mu * cs + mu * cs;
    if (det == 0.0) {
        // No s-waves
        a3 = 0.0;
        a4 = 0.0;
    } else {
        a3 = (cs * dsig12 + mu * dv) / det;
        a4 = (cs * dsig12 - mu * dv) / det;
    }

    // Compute the waves
    // First wave (P-wave, left-going)
    wave[0][0] = a1 * bulkl;  // sigma11
    wave[0][1] = a1 * lambda; // sigma22
    wave[0][2] = 0.0;         // sigma12
    wave[0][3] = a1 * cp;     // u
    wave[0][4] = 0.0;         // v
    s[0] = -cp;

    // Second wave (P-wave, right-going)
    wave[1][0] = a2 * bulkr;
    wave[1][1] = a2 * lambda;
    wave[1][2] = 0.0;
    wave[1][3] = -a2 * cp;
    wave[1][4] = 0.0;
    s[1] = cp;

    // Third wave (S-wave, left-going)
    wave[2][0] = 0.0;
    wave[2][1] = 0.0;
    wave[2][2] = a3 * mu;
    wave[2][3] = 0.0;
    wave[2][4] = a3 * cs;
    s[2] = -cs;

    // Fourth wave (S-wave, right-going)
    wave[3][0] = 0.0;
    wave[3][1] = 0.0;
    wave[3][2] = a4 * mu;
    wave[3][3] = 0.0;
    wave[3][4] = -a4 * cs;
    s[3] = cs;

    // Compute flux differences (amdq, apdq)
    for (PetscInt m = 0; m < Nf; m++) {
        flux[m] = s[1] * wave[1][m] + s[3] * wave[3][m] - s[0] * wave[0][m] - s[2] * wave[2][m];
    }
}

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
  char      outputBasename[PETSC_MAX_PATH_LEN]; /* Basename for output files */
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
  PetscPrintf(PETSC_COMM_WORLD, "dim = %d, Nc = %d, time = %f, x = (%f, %f)\n", dim, Nc, time, x[0], x[1]);
  u[0] = 0.0;
  u[1] = 0.0;
  u[2] = 0.0;
  u[3] = 0.0;
  u[4] = 0.0;

  return PETSC_SUCCESS;
}

static PetscErrorCode SetUpBC(DM dm, PetscDS ds, Physics phys)
{
  DMLabel        label;
  PetscInt       field = 0;   // we're working with a single field
  
  PetscFunctionBeginUser;
  /* Add Dirichlet boundary conditions
     PetscDSAddBoundary:
     
     Input Parameters:
       ds       - The PetscDS object
       type     - The type of condition, e.g. `DM_BC_ESSENTIAL`/`DM_BC_ESSENTIAL_FIELD` (Dirichlet), or `DM_BC_NATURAL` (Neumann)
       name     - The BC name
       label    - The label defining constrained points
       Nv       - The number of `DMLabel` values for constrained points
       values   - An array of label values for constrained points
       field    - The field to constrain
       Nc       - The number of constrained field components (0 will constrain all fields)
       comps    - An array of constrained component numbers
       bcFunc   - A pointwise function giving boundary values
       bcFunc_t - A pointwise function giving the time derivative of the boundary values, or NULL
       ctx      - An optional user context for bcFunc
    
     Output Parameter:
       bd - The boundary number
    
     Options Database Keys:
       -bc_<boundary name> <num>      - Overrides the boundary ids
       -bc_<boundary name>_comp <num> - Overrides the boundary components
  */

  PetscInt boundaryids[] = {4};  // Physical group label for "Left"

  PetscCall(DMGetLabel(dm, "boundary", &label));
  PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "boundary", label, 1, boundaryids,
                     field, 0, NULL, (void (*)(void))ZeroBoundaryCondition,
                     NULL, NULL, NULL);
  PetscCall(DMViewFromOptions(dm, NULL, "-after_ds"));
  PetscDSViewFromOptions(ds, NULL, "-ds_view");
  //PetscCall(DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD));
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
  for (d = 0; d < dim; ++d) {
    u[d] = 0.0;
  }
  
 // if (PetscAbsReal(x[0]) < 0.5 && PetscAbsReal(x[1]) < 0.5) {
 //   u[0] = 1;  // Apply source to the first field (modify as needed for other fields)
 // }


  // Define a small area in the center (e.g., radius r = 0.1 around the point (0.5, 0.5))
  PetscReal radius = 0.1;
  PetscReal center_x = 0.5;
  PetscReal center_y = 0.5;
  PetscReal distance_from_center = PetscSqrtReal((x[0] - center_x)*(x[0] - center_x) + (x[1] - center_y)*(x[1] - center_y));

  // Check if the point is within the small area around the center (0.5, 0.5)
  if (distance_from_center < radius) {
    u[0] = 1.0;  // Set u[0] to 1 in the central region
  }
  
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
  char        filename[PETSC_MAX_PATH_LEN];
  PetscReal   xnorm;
  PetscBool   rollback;

  PetscFunctionBeginUser;
  // Check for rollback
  PetscCall(TSGetStepRollBack(ts, &rollback));
  if (rollback)
    PetscFunctionReturn(PETSC_SUCCESS);

  // Get the current solution
  PetscCall(PetscObjectSetName((PetscObject)X, "u"));
  // Get the DM associated with the solution vector X
  PetscCall(VecGetDM(X, &dm));
  // Find the norm of the solution vector for summary printing
  PetscCall(VecNorm(X, NORM_INFINITY, &xnorm));

  // Adjust step iteration number by user offset
  if (stepnum >= 0) stepnum += user->monitorStepOffset;

  // Process and print results (omit if stepnum = -1, i.e., final time)
  if (stepnum >= 0) {
    Vec                cellgeom;
    PetscInt           c, cStart, cEnd;
    const PetscScalar *cgeom, *x;
    DM                 dmCell;

    // Ensure the DM is DMPlex
    PetscCall(DMConvert(dm, DMPLEX, &plex));
    // Get the FV mesh geometry (optional, depending on your needs)
    PetscCall(DMPlexGetGeometryFVM(plex, NULL, &cellgeom, NULL));

    // Get the DM associated with the FV cell geometry (optional)
    PetscCall(VecGetDM(cellgeom, &dmCell));
    // Get the range of cells in the mesh
    PetscCall(DMPlexGetSimplexOrBoxCells(dmCell, 0, &cStart, &cEnd));

    // Read the arrays (geometry of cells and solution)
    PetscCall(VecGetArrayRead(cellgeom, &cgeom));  // optional
    PetscCall(VecGetArrayRead(X, &x));

    // Loop over all cells and directly output field values to VTK
    for (c = cStart; c < cEnd; ++c) {
      const PetscScalar *cx = NULL;

      // Read the solution at the current cell
      PetscCall(DMPlexPointGlobalRead(dm, c, x, &cx));

      if (!cx) continue;  // Skip ghost or non-global cells

      // Here, you can process cx directly to output its components at the cell
      // In this case, simply write the components of `cx` to VTK
      // You can expand this part with code that writes to the VTK file
    }

    // Restore the arrays
    PetscCall(VecRestoreArrayRead(cellgeom, &cgeom));  // optional
    PetscCall(VecRestoreArrayRead(X, &x));
    PetscCall(DMDestroy(&plex));
  }

  // Output to VTK at regular intervals or at the final time
  if (user->vtkInterval < 1) PetscFunctionReturn(PETSC_SUCCESS);
  if ((stepnum == -1) ^ (stepnum % user->vtkInterval == 0)) {
    if (stepnum == -1) {
      PetscCall(TSGetStepNumber(ts, &stepnum));  // Adjust for final time
    }
    PetscCall(PetscSNPrintf(filename, sizeof(filename), "%s-%03" PetscInt_FMT ".vtu", user->outputBasename, stepnum));
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
  //PetscCall(TSMonitorSet(*ts, MonitorVTK, user, NULL));
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
  PetscViewer       viewer;

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
  user->vtkmon = PETSC_TRUE;
  user->vtkInterval = 1;
  PetscStrcpy(user->outputBasename, "paraview");

  // Process user options
  PetscCall(ProcessOptions(comm, user));
  
  // Read in 3D mesh from file
  //PetscCall(DMPlexCreateFromFile(comm, user->infile, NULL, PETSC_TRUE,&dm));
  //PetscCall(DMPlexCreateGmshFromFile(comm, user->infile, PETSC_TRUE, &dm));
//  PetscViewerBinaryOpen(comm, "mesh.msh", FILE_MODE_READ, &viewer);
  PetscCall(PetscViewerCreate(comm,&viewer));
  PetscCall(PetscViewerSetType(viewer,PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(viewer,"mesh.msh"));
  DMPlexCreateGmsh(comm, viewer, PETSC_TRUE, &dm);
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-orig_dm_view"));
  PetscCall(DMGetDimension(dm, &dim));
  // create label for boundary conditions
  PetscCall(DMCreateLabel(dm, "Face Sets"));

  {
    DM gdm;

    PetscCall(DMPlexConstructGhostCells(dm, NULL, NULL, &gdm));
    PetscCall(DMDestroy(&dm));
    dm = gdm;
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  }

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
  //  PetscCall(PetscDSSetContext(ds, 0, user->model->physics));
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
  PetscCall(VecViewFromOptions(X, NULL, "-X_view"));
  PetscCall(VecDestroy(&X));
  PetscCall(TSViewFromOptions(ts, NULL, "-ts_view"));
  PetscCall(PetscDSViewFromOptions(ds, NULL, "-ds_view"));
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

