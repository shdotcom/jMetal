//  RealGenerationalGeneticAlgorithmRunner.java
//
//  Author:
//       Antonio J. Nebro <antonio@lcc.uma.es>
//
//  Copyright (c) 201r Antonio J. Nebro
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
// 
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

package org.uma.jmetal.runner.singleobjective;

import org.uma.jmetal.core.Algorithm;
import org.uma.jmetal.core.Problem;
import org.uma.jmetal.core.SolutionSet;
import org.uma.jmetal.metaheuristic.singleobjective.geneticalgorithm.GenerationalGeneticAlgorithm;
import org.uma.jmetal.operator.crossover.Crossover;
import org.uma.jmetal.operator.crossover.SBXCrossover;
import org.uma.jmetal.operator.mutation.Mutation;
import org.uma.jmetal.operator.mutation.PolynomialMutation;
import org.uma.jmetal.operator.selection.BinaryTournament;
import org.uma.jmetal.operator.selection.Selection;
import org.uma.jmetal.problem.singleobjective.Sphere;
import org.uma.jmetal.util.AlgorithmRunner;
import org.uma.jmetal.util.JMetalException;
import org.uma.jmetal.util.JMetalLogger;
import org.uma.jmetal.util.fileOutput.DefaultFileOutputContext;
import org.uma.jmetal.util.fileOutput.SolutionSetOutput;

import java.io.IOException;

/**
 * This class runs a single-objective genetic algorithm (GA). The GA can be
 * a steady-state GA (class SteadyStateGeneticAlgorithm), a generational GA (class gGA), a synchronous
 * cGA (class SynchronousCellularGA) or an asynchronous cGA (class AsynchronousCellularGA). The OneMax
 * problem is used to org.uma.test the algorithms.
 */
public class RealGenerationalGeneticAlgorithmRunner {

  public static void main(String[] args) throws JMetalException, ClassNotFoundException, IOException {
    Problem problem;
    Algorithm algorithm;
    Crossover crossover;
    Mutation mutation;
    Selection selection;

    problem = new Sphere("Real", 30);

    crossover = new SBXCrossover.Builder()
            .probability(0.9)
            .distributionIndex(20)
            .build() ;

    mutation = new PolynomialMutation.Builder()
            .probability(1.0/problem.getNumberOfVariables())
            .distributionIndex(20)
            .build() ;

    selection = new BinaryTournament.Builder()
            .build() ;

    algorithm = new GenerationalGeneticAlgorithm.Builder(problem)
            .setPopulationSize(100)
            .setMaxEvaluations(250000)
            .setCrossover(crossover)
            .setMutation(mutation)
            .setSelection(selection)
            .build() ;

    AlgorithmRunner algorithmRunner = new AlgorithmRunner.Executor(algorithm)
            .execute() ;

    SolutionSet population = algorithmRunner.getSolutionSet() ;
    long computingTime = algorithmRunner.getComputingTime() ;

    new SolutionSetOutput.Printer(population)
            .separator("\t")
            .varFileOutputContext(new DefaultFileOutputContext("VAR.tsv"))
            .funFileOutputContext(new DefaultFileOutputContext("FUN.tsv"))
            .print();

    JMetalLogger.logger.info("Total execution time: " + computingTime + "ms");
    JMetalLogger.logger.info("Objectives values have been written to file FUN.tsv");
    JMetalLogger.logger.info("Variables values have been written to file VAR.tsv");
  }
}
