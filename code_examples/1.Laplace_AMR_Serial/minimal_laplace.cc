/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */

// Minimal non-trivial example:
// - Laplacian (conforming) on a L-shaped domain using Meshworker::mesh_loop()
// - Adaptive mesh refinement

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;


template <int dim, int spacedim>
void
run(const unsigned int n_refinement_cycles, const unsigned int fe_degree);


int
main()
{
  constexpr int dim = 2;
  constexpr int spacedim = 2;
  constexpr unsigned int n_refinement_cycles = 8;
  constexpr unsigned int fe_degree = 1;

  run<dim, spacedim>(n_refinement_cycles, fe_degree);
}


template <int dim, int spacedim>
void
run(const unsigned int n_refinement_cycles, const unsigned int fe_degree)
{
  const FE_Q<dim, spacedim>    fe(fe_degree);
  const QGauss<dim>     cell_quadrature(fe.degree+1);
  const QGauss<dim - 1> face_quadrature(fe.degree+1);

  Triangulation<dim, spacedim> tria;
  DoFHandler<dim, spacedim>    dof_handler(tria);

  const ConstantFunction<spacedim> rhs_function(1);
  const ZeroFunction<spacedim> boundary_function;

  AffineConstraints<double> constraints;
  Vector<double> solution;

  for (unsigned int cycle=0; cycle < n_refinement_cycles; ++cycle)
  {
    if (cycle == 0)
    {
      GridGenerator::hyper_L(tria);
      tria.refine_global(1);
    }
    else
    {
      Vector<float> estimated_error_per_cell(tria.n_active_cells());

      KellyErrorEstimator<dim>::estimate(
        dof_handler,
        QGauss<dim - 1>(fe.degree + 1),
        std::map<types::boundary_id, const Function<dim> *>{{0,&boundary_function}},
        solution,
        estimated_error_per_cell);

      GridRefinement::refine_and_coarsen_fixed_number(tria,
                                                      estimated_error_per_cell,
                                                      0.3,
                                                      0.03);
      tria.execute_coarsening_and_refinement();
    }

    dof_handler.distribute_dofs(fe);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             boundary_function,
                                             constraints);

    constraints.close();

    SparsityPattern sparsity;
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      sparsity.copy_from(dsp);
    }

    SparseMatrix<double> system_matrix;
    system_matrix.reinit(sparsity);

    Vector<double> system_rhs(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());

    const UpdateFlags cell_flags = update_values | update_gradients |
                                   update_quadrature_points | update_JxW_values;

    using ScratchData = MeshWorker::ScratchData<dim, spacedim>;
    using CopyData    = MeshWorker::CopyData<1, 1, 1>;

    ScratchData scratch(fe, cell_quadrature, cell_flags);
    CopyData    copy(fe.dofs_per_cell);

    using Iterator = decltype(dof_handler.begin_active());

    auto cell_worker =
      [&rhs_function](const Iterator &cell, ScratchData &scratch_data, CopyData &copy_data) {
        const auto &fe_values = scratch_data.reinit(cell);
        const auto &JxW       = scratch_data.get_JxW_values();
        const auto &q_points  = scratch_data.get_quadrature_points();

        copy_data = 0;
        copy_data.local_dof_indices[0] = scratch_data.get_local_dof_indices();

        for (unsigned int q = 0; q < q_points.size(); ++q)
          for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < fe_values.dofs_per_cell; ++j)
                {
                  copy_data.matrices[0](i, j) +=
                    fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * JxW[q];
                }
              copy_data.vectors[0](i) +=
                fe_values.shape_value(i, q) * rhs_function.value(q_points[q]) * JxW[q];
            }
      };

    auto copier = [&constraints, &system_matrix, &system_rhs](const CopyData &copy_data) {
      constraints.distribute_local_to_global(
        copy_data.matrices[0], copy_data.vectors[0], 
        copy_data.local_dof_indices[0], 
        system_matrix, system_rhs);
    };

    MeshWorker::mesh_loop(dof_handler.active_cell_iterators(), cell_worker, 
                          copier, scratch, copy, 
                          MeshWorker::assemble_own_cells);

    SolverControl solver_control(system_matrix.m(), 1e-12);
    SolverCG<>    solver(solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(fe_degree);

    std::ofstream output("solution-" + std::to_string(dim) + "d-" +
                         std::to_string(cycle) + ".vtu");
    data_out.write_vtu(output);
  }
}
