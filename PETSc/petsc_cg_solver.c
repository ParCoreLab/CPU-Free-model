static char help[] =
    "Solves a linear system in parallel with KSP.\n\
Input parameters include:\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_y>       : number of mesh points in y-direction\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.
*/
#include <petscksp.h>

int main(int argc, char **args) {
    Vec x, b;      /* approx solution, RHS */
    Mat A;         /* linear system matrix */
    KSP cg_solver; /* linear solver context */
    PC preconditioner;
    PetscInt num_rows, num_cols, its;

    PetscViewer viewer;
    char matrix_filename[PETSC_MAX_PATH_LEN] = "";

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-matrix_path", matrix_filename,
                                    sizeof(matrix_filename), NULL));

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, matrix_filename, FILE_MODE_READ, &viewer));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Compute the matrix and right-hand-side vector that define
           the linear system, Ax = b.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
       Create parallel matrix, specifying only its global dimensions.
       When using MatCreate(), the matrix format can be specified at
       runtime. Also, the parallel partitioning of the matrix is
       determined by PETSc at runtime.

       Performance tuning note:  For problems of substantial size,
       preallocation of matrix memory is crucial for attaining good
       performance. See the matrix chapter of the users manual for details.
    */
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetType(A, MATMPIAIJCUSPARSE));
    PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatLoad(A, viewer));
    PetscCall(MatGetLocalSize(A, &num_rows, &num_cols));

    PetscCall(MatCreateVecs(A, &x, &b));
    PetscCall(VecSet(b, 1.0));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                  Create the linear solver and set various options
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &cg_solver));
    PetscCall(KSPSetOperators(cg_solver, A, A));
    PetscCall(KSPSetType(cg_solver, KSPCG));
    PetscCall(KSPSetInitialGuessNonzero(cg_solver, PETSC_FALSE));
    PetscCall(KSPGetPC(cg_solver, &preconditioner));
    PetscCall(PCSetType(preconditioner, PCNONE));
    PetscCall(KSPSetTolerances(cg_solver, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 5000));
    PetscCall(KSPSetNormType(cg_solver, KSP_NORM_NONE));
    PetscCall(KSPSetConvergenceTest(cg_solver, KSPConvergedSkip, NULL, NULL));

    /*
      Set runtime options, e.g.,
          -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
      These options will override those specified above as long as
      KSPSetFromOptions() is called _after_ any other customization
      routines.
    */
    PetscCall(KSPSetFromOptions(cg_solver));
    PetscCall(KSPSetUp(cg_solver));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Solve the linear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    PetscCall(KSPSolve(cg_solver, b, x));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Check the solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(KSPGetIterationNumber(cg_solver, &its));

    /*
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
    */
    PetscCall(KSPDestroy(&cg_solver));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));

    /*
       Always call PetscFinalize() before exiting a program.  This routine
         - finalizes the PETSc libraries as well as MPI
         - provides summary and diagnostic information if certain runtime
           options are chosen (e.g., -log_view).
    */
    PetscCall(PetscFinalize());
    return 0;
}