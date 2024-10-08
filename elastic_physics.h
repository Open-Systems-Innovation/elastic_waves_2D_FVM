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
