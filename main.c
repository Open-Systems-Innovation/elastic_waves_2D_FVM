#include "petscsystypes.h"
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
  //PetscPrintf(PETSC_COMM_WORLD, "dim = %d, Nf = % d, qp = (%f, %f), n = (%f, %f), \n uL =  (%f, %f, %f, %f, %f), \n uR = (%f, %f, %f, %f, %f), numConstants = %d\n", dim, Nf, qp[0], qp[1], n[0], n[1], uL[0], uL[1], uL[2], uL[3], uL[4], uR[0], uR[1], uR[2], uR[3], uR[4], numConstants );

  PetscReal du[5];
  PetscReal detP, detS;
  PetscReal a1, a2, a3, a4; // alpha values
  
  // Compute jumps in the states (uL is left, uR is right)
  // and store in du
  for (PetscInt i = 0; i < 5; ++i) {
      du[i] = uR[i] - uL[i];
  }

  // obtain material paramter values for L and R sides
  PetscReal lambdaL = uL[5];
  PetscReal muL = uL[6];
  PetscReal densityL = uL[7];
  PetscReal lambdaR = uR[5];
  PetscReal muR = uR[6];
  PetscReal densityR = uR[7];
  
  // Calculate cp (P-wave speed) and cs (S-wave speed)
  PetscReal bulkL = lambdaL + 2.0 * muL;
  PetscReal bulkR = lambdaR + 2.0 * muR;
  PetscReal cpL = PetscSqrtReal(bulkL / densityL);
  PetscReal csL = PetscSqrtReal(muL / densityL);
  PetscReal cpR = PetscSqrtReal(bulkR / densityR);
  PetscReal csR = PetscSqrtReal(muR / densityR);

  //PetscPrintf(PETSC_COMM_WORLD, "Right P-wave speed (cp): %f\n", cpR);
  //PetscPrintf(PETSC_COMM_WORLD, "Right S-wave speed (cs): %f\n", csR);
  //PetscPrintf(PETSC_COMM_WORLD, "Left P-wave speed (cp): %f\n", cpL);
  //PetscPrintf(PETSC_COMM_WORLD, "Left S-wave speed (cs): %f\n", csL);

  // Calculate useful multiples of the norm vector components
  PetscReal nx = n[0];
  PetscReal ny = n[1];
  PetscReal nx2 = nx * nx;
  PetscReal ny2 = ny * ny;
  PetscReal nxy = nx * ny;
  
  // Define Eigenvalues (wave speeds)
  PetscReal s[5] = {-cpL, cpR, -csL, csR, 0};

  // Define the 4 eigenvectors (from columns of Matrix R)
  PetscReal r1[5] = {lambdaL + 2 * muL * nx2, lambdaL + 2 * muL * ny2, 2 * muL * nxy,
                     nx * cpL, ny * cpL};
  PetscReal r2[5] = {lambdaR + 2 * muR * nx2, lambdaR + 2 * muR * ny2, 2 * muR * nxy,
                     -nx * cpR, -ny * cpR};
  PetscReal r3[5] = {-2 * muL * nxy, 2 * muL * nxy, muL * (nx2 - ny2),
                     -ny * csL, nx * csL};
  PetscReal r4[5] = {-2 * muR * nxy, 2 * muR * nxy, muR * (nx2 - ny2), ny * csR,
                     -nx * csR};
  PetscReal r5[5] = {ny2, nx2, -nxy, 0, 0}; // this one doesn't add anything

  // Compute the 4 alphas
  detP = cpR * bulkL + cpL * bulkR;
  detS = csR * muL + csL * muR;

  // P wave strengths
  a1 = (cpR * (du[0] * nx2 + du[1] * ny2 + 2 * nxy * du[2]) +
         bulkR * (nx * du[3] + ny * du[4])) / detP;
  a2 = (cpL * (du[0] * nx2 + du[1] * ny2 + 2 * nxy * du[2]) -
         bulkL * (nx * du[3] + ny * du[4])) / detP;
  // S wave strengths
  a3 = (csR * (du[2] * (nx2 - ny2) + nxy * (du[1] - du[0])) +
         muR * (nx * du[4] - ny * du[3])) / detS;
  a4 = (csL * (du[2] * (nx2 - ny2) + nxy * (du[1] - du[0])) -
         muL * (nx * du[4] - ny * du[3])) / detS;
 
  // Compute the waves
// Compute the waves
  PetscReal W1[5], W2[5], W3[5], W4[5];
  for (int i = 0; i < 5; ++i) {
      W1[i] = a1 * r1[i];
      W2[i] = a2 * r2[i];
      W3[i] = a3 * r3[i];
      W4[i] = a4 * r4[i];
  }

  // First wave (P-wave, left-going)
  // Second wave (P-wave, right-going)
  // Third wave (S-wave, left-going)
  // Fourth wave (S-wave, right-going)

  // Use the waves to update the flux
  flux[0] = s[1] * W2[0] + s[3] * W4[0] - s[0] * W1[0] - s[2] * W3[0];
  flux[1] = s[1] * W2[1] + s[3] * W4[1] - s[0] * W1[1] - s[2] * W3[1];
  flux[2] = s[1] * W2[2] + s[3] * W4[2] - s[0] * W1[2] - s[2] * W3[2];
  flux[3] = s[1] * W2[3] + s[3] * W4[3] - s[0] * W1[3] - s[2] * W3[3];
  flux[4] = s[1] * W2[4] + s[3] * W4[4] - s[0] * W1[4] - s[2] * W3[4];
  flux[5] = 0.0;
  flux[6] = 0.0;
  flux[7] = 0.0;
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
  PetscReal maxspeed;
  Model     model;
  PetscInt  monitorStepOffset;
  char      outputBasename[PETSC_MAX_PATH_LEN]; /* Basename for output files */
  PetscInt  vtkInterval;                        /* For monitor */
  PetscBool vtkmon;
};

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Boundary Conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
static PetscErrorCode BoundaryOutflow(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *xI, PetscScalar *xG, void *ctx)
{
  PetscFunctionBeginUser;
  //PetscPrintf(PETSC_COMM_WORLD, "time= %f, c = (%f, %f), n = (%f, %f), xI = (%f, %f, %f, %f, %f),  xG = (%f, %f)\n", time, c[0], c[1], n[0], n[1], xI[0], xI[1], xI[2], xI[3], xI[4], xG[0], xG[1]);
  xG[0] = xI[0];
  xG[1] = xI[1];
  xG[2] = xI[2];
  xG[3] = xI[3];
  xG[4] = xI[4];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetUpBC(DM dm, PetscDS ds, Physics phys)
{
  DMLabel        label;
  PetscInt       boundaryid=1;  // Physical group label for "boundary"
  
  PetscFunctionBeginUser;
  
  PetscCall(PetscDSViewFromOptions(ds, NULL, "-ds_view"));
  // Add Dirichlet boundary conditions
  // Check if the label exists
  PetscCall(DMGetLabel(dm, "boundary", &label));
  // if it doesn't exist, then throw error
  if (!label) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: Label 'boundary' not found\n"));
      PetscFunctionReturn(PETSC_ERR_ARG_WRONG);
  }
  
  PetscCall(PetscDSAddBoundary(ds, DM_BC_NATURAL_RIEMANN, "boundary", label, 1, &boundaryid, 0, 0, NULL, (void (*)(void))BoundaryOutflow, NULL, phys, NULL));

  PetscCall(DMViewFromOptions(dm, NULL, "-after_ds"));
  //PetscCall(DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode transmitterMaterial (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *modctx)
{
  PetscFunctionBeginUser;
  u[0] = 1;
  u[1] = 1;
  u[2] = 1;
  u[3] = 1;
  u[4] = 1;
  u[5] = 1;
  u[6] = 1;
  u[7] = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode blobMaterial (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *modctx)
{
  PetscFunctionBeginUser;
  u[0] = 0;
  u[1] = 0;
  u[2] = 0;
  u[3] = 0;
  u[4] = 0;
  u[5] = 5;
  u[6] = 5;
  u[7] = 10;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode siliconeMaterial (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *modctx)
{
  PetscFunctionBeginUser;
  u[0] = 0;
  u[1] = 0;
  u[2] = 0;
  u[3] = 0;
  u[4] = 0;
  u[5] = 1;
  u[6] = 1;
  u[7] = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetUpMaterialProperties(DM dm, Vec X)
{
  DMLabel label;
  PetscInt Nc = 8;
  PetscInt materialid; 
  
  // declare an array of function pointers called func
  // one function per field
  // we have to do it like this becaues DMProjectFunction expects an
  // array of function pointers because C is like that
  PetscErrorCode (*func[1])(PetscInt dim, PetscReal time, const PetscReal x[],
                            PetscInt Nf, PetscScalar *u, void *ctx);
  // now func[0] is a pointer to a function

  PetscFunctionBeginUser;

  // transmitter material
  materialid = 1;
  func[0] = transmitterMaterial;
  PetscCall(DMGetLabel(dm, "tramsmitter", &label));
  PetscCall(DMProjectFunctionLabel(dm, 0.0, label, 1, &materialid, Nc, NULL, func, NULL, INSERT_ALL_VALUES, X));
  // blob material
  materialid = 2;
  func[0] = blobMaterial;
  PetscCall(DMGetLabel(dm, "blob", &label));
  PetscCall(DMProjectFunctionLabel(dm, 0.0, label, 1, &materialid, Nc, NULL, func, NULL, INSERT_ALL_VALUES, X));
  // silicone material
  materialid = 3;
  func[0] = siliconeMaterial;
  PetscCall(DMGetLabel(dm, "silicone", &label));
  PetscCall(DMProjectFunctionLabel(dm, 0.0, label, 1, &materialid, Nc, NULL, func, NULL, INSERT_ALL_VALUES, X));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Initial Conditions 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
PetscErrorCode zero_vector(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  
  // Initialize all values to zero initially
  for (PetscInt d = 0; d < 5; ++d) {
    u[d] = 0.0;
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
  DM          dm;
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
  PetscCall(VecGetDM(X, &dm));

  // Find the norm of the solution vector for summary printing
  PetscCall(VecNorm(X, NORM_INFINITY, &xnorm));

  // Adjust step iteration number by user offset
  if (stepnum >= 0) stepnum += user->monitorStepOffset;

  // Output to VTK at regular intervals or at the final time
  if (user->vtkInterval < 1) PetscFunctionReturn(PETSC_SUCCESS);
  if ((stepnum == -1) ^ (stepnum % user->vtkInterval == 0)) {
    if (stepnum == -1) {
      PetscCall(TSGetStepNumber(ts, &stepnum));  // Adjust for final time
    }

    // Generate the VTK filename and open the VTK viewer
    PetscCall(PetscSNPrintf(filename, sizeof(filename), "%s-%03" PetscInt_FMT ".vtu", user->outputBasename, stepnum));
    PetscCall(OutputVTK(dm, filename, &viewer));

    // Write the solution vector to VTK using VecView
    PetscCall(VecView(X, viewer));

    // Clean up
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
  PetscReal         ftime, cfl, dt, minRadius, maxspeed;
  PetscInt          nsteps;
  PetscInt          dim = 2;
  PetscInt          numComponents = 8;
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
  PetscCall(PetscViewerFileSetName(viewer,"rectangle_with_circle.msh"));
  PetscCall(DMPlexCreateGmsh(comm, viewer, PETSC_TRUE, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-orig_dm_view"));
  PetscCall(DMGetDimension(dm, &dim));
  // create label for boundary conditions
  //PetscCall(DMCreateLabel(dm, "Face Sets"));

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
  PetscCall(PetscObjectSetName((PetscObject)fv, "finite_volume"));
  
  // Define component names for pressure and velocity
  {
    // Set names for the components of the field 
    PetscCall(PetscFVSetComponentName(fv, 0, "Stress_11")); // 12 component of stress 
    PetscCall(PetscFVSetComponentName(fv, 1, "Stress_22")); // 22 component of stress 
    PetscCall(PetscFVSetComponentName(fv, 2, "Stress_12")); // 12 component of stress 
    PetscCall(PetscFVSetComponentName(fv, 3, "Velocity_x")); // X component of velocity 
    PetscCall(PetscFVSetComponentName(fv, 4, "Velocity_y")); // Y component of velocity
    PetscCall(PetscFVSetComponentName(fv, 5, "Lambda")); // Y component of velocity
    PetscCall(PetscFVSetComponentName(fv, 6, "Mu")); // Y component of velocity
    PetscCall(PetscFVSetComponentName(fv, 7, "Density")); // Y component of velocity
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
  //PetscCall(SetInitialConditions(dm, X, user));
  PetscCall(SetUpMaterialProperties(dm, X));
  
  PetscCall(DMPlexGetGeometryFVM(dm, NULL, NULL, &minRadius));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  //PetscCall(DMDestroy(&dm));
  //PetscCall(MPIU_Allreduce(&phys->maxspeed, &mod->maxspeed, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)ts)));
  
  cfl = 0.9 * 4; /* default SSPRKS2 with s=5 stages is stable for CFL number s-1 */
  maxspeed = 100;
  //dt = 0.01;
  dt = cfl * minRadius / maxspeed;
  PetscPrintf(comm, "dt = %f", dt);
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

