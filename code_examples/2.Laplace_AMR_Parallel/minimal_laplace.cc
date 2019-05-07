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
// - Parallel distributed triangulation

#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

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

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

using namespace dealii;

template <int dim, int spacedim>
void
run(MPI_Comm mpi_communicator, const unsigned int n_refinement_cycles, const unsigned int fe_degree);


int
main(int argc, char *argv[])
{
  constexpr int dim = 2;
  constexpr int spacedim = 2;
  constexpr unsigned int n_refinement_cycles = 8;
  constexpr unsigned int fe_degree = 1;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  MPI_Comm mpi_communicator (MPI_COMM_WORLD);

  run<dim, spacedim>(mpi_communicator, n_refinement_cycles, fe_degree);
}


template <int dim, int spacedim>
void
run(MPI_Comm mpi_communicator, const unsigned int n_refinement_cycles, const unsigned int fe_degree)
{
  const FE_Q<dim, spacedim>    fe(fe_degree);
  const QGauss<dim>     cell_quadrature(fe.degree+1);
  const QGauss<dim - 1> face_quadrature(fe.degree+1);

  parallel::distributed::Triangulation<dim, spacedim> tria (mpi_communicator);
  DoFHandler<dim, spacedim> dof_handler(tria);

  const ConstantFunction<spacedim> rhs_function(1);
  const ZeroFunction<spacedim> boundary_function;

  AffineConstraints<double> constraints;
  TrilinosWrappers::MPI::Vector locally_relevant_solution;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

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
        std::map<types::boundary_id, const Function<dim> *>(),
        //std::map<types::boundary_id, const Function<dim> *>{{0,&boundary_function}},
        locally_relevant_solution,
        estimated_error_per_cell);

      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria, estimated_error_per_cell,
        0.3, 0.03);
      tria.execute_coarsening_and_refinement();
    }

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    constraints.clear();
    // constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             boundary_function,
                                             constraints);

    constraints.close();

    TrilinosWrappers::SparseMatrix system_matrix;
    {
      DynamicSparsityPattern dsp(locally_relevant_dofs);

      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(
        dsp,
        dof_handler.n_locally_owned_dofs_per_processor(),
        mpi_communicator,
        locally_relevant_dofs);

      system_matrix.reinit(locally_owned_dofs,
                           locally_owned_dofs,
                           dsp,
                           mpi_communicator);
    }

    TrilinosWrappers::MPI::Vector system_rhs(locally_owned_dofs, mpi_communicator);
    locally_relevant_solution.reinit(
      locally_owned_dofs,
      locally_relevant_dofs,
      mpi_communicator);

    const UpdateFlags cell_flags = update_values | update_gradients |
                                   update_quadrature_points | update_JxW_values;

    using ScratchData = MeshWorker::ScratchData<dim, spacedim>;
    using CopyData    = MeshWorker::CopyData<1, 1, 1>;

    ScratchData scratch(fe, cell_quadrature, cell_flags);
    CopyData    copy(fe.dofs_per_cell);

    auto cell = dof_handler.begin_active();
    const auto endc = dof_handler.end();

    using Iterator = decltype(cell);

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

    MeshWorker::mesh_loop(cell, endc, cell_worker, 
                          copier, scratch, copy, 
                          MeshWorker::assemble_own_cells);

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    SolverControl solver_control(system_matrix.m(), 1e-12);
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
    TrilinosWrappers::PreconditionAMG preconditioner;
    TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
    additional_data.higher_order_elements = (fe.degree > 1);
    preconditioner.initialize(system_matrix, additional_data);

    TrilinosWrappers::MPI::Vector distributed_solution (locally_owned_dofs, mpi_communicator);
    solver.solve(system_matrix, distributed_solution, system_rhs, preconditioner);
    constraints.distribute(distributed_solution);
    locally_relevant_solution = distributed_solution;

    Vector<float> subdomain(tria.n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = tria.locally_owned_subdomain();

    DataOut<dim> data_out;
    DataOutBase::VtkFlags output_flags;
    output_flags.write_higher_order_cells = true;
    data_out.set_flags(output_flags);
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "solution");
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches(fe.degree);

    const std::string filename("solution-" + std::to_string(dim) + "d-" +
                               std::to_string(cycle) + ".vtu");
    data_out.write_vtu_in_parallel(filename, mpi_communicator);
  }
}
