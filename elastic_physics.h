#include <petscdm.h>

#define DIM 2 /* Geometric dimension */

/* Represents continuum physical equations. */
typedef struct _n_Physics *Physics;

/* Physical model includes boundary conditions, initial conditions, and functionals of interest. It is
 * discretization-independent, but its members depend on the scenario being solved. */
typedef struct _n_Model *Model;

struct FieldDescription {
  const char *name;
  PetscInt    dof;
};

struct _n_Physics {
  void (*riemann)(PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscScalar[], const PetscScalar[], PetscInt, const PetscScalar[], PetscScalar[], void *);
  PetscInt                       dof;      /* number of degrees of freedom per cell */
  PetscReal                      maxspeed; /* kludge to pick initial time step, need to add monitoring and step control */

  void                          *data;
  PetscInt                       nfields;
  const struct FieldDescription *field_desc;
};

typedef struct {
  PetscReal density;
  PetscReal lame_1;
  PetscReal lame_2;
  struct {
    PetscInt Density;
    PetscInt Momentum;
    PetscInt Energy;
    PetscInt Pressure;
    PetscInt Speed;
  } monitor;
} Physics_Elastic;

typedef struct {
  PetscReal sigma_11;
  PetscReal sigma_22;
  PetscReal sigma_12;
  PetscReal v_x;
  PetscReal v_y;
} ElasticNode;

typedef union
{
  ElasticNode eulernode;
  PetscReal vals[DIM + 2];
} ElasticNodeUnion;

typedef struct {
  PetscReal rho;    // Density
  PetscReal lambda; // Lambda parameter
  PetscReal mu;     // Mu parameter
  PetscReal cp;     // P-wave speed
  PetscReal cs;     // S-wave speed
} ElasticityContext;

static void Elastic_Riemann_Godunov(
    PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n,
    const PetscScalar *uL, const PetscScalar *uR, PetscInt numConstants,
    const PetscScalar constants[], PetscScalar *flux, void *ctx)
{
  ElasticityContext *material = (ElasticityContext*) ctx; // Cast user context

    PetscReal rho = material->rho;
    PetscReal lambda = material->lambda;
    PetscReal mu = material->mu;
    PetscReal cp = material->cp;
    PetscReal cs = material->cs;

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

