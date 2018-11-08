package org.uma.jmetal.algorithm.multiobjective.rnsgaii;

import org.uma.jmetal.algorithm.InteractiveAlgorithm;
import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAII;
import org.uma.jmetal.measure.Measurable;
import org.uma.jmetal.measure.MeasureManager;
import org.uma.jmetal.measure.impl.BasicMeasure;
import org.uma.jmetal.measure.impl.CountingMeasure;
import org.uma.jmetal.measure.impl.DurationMeasure;
import org.uma.jmetal.measure.impl.SimpleMeasureManager;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.SelectionOperator;
import org.uma.jmetal.operator.impl.selection.RankingAndPreferenceSelection;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.util.comparator.DominanceComparator;
import org.uma.jmetal.util.evaluator.SolutionListEvaluator;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Antonio J. Nebro <antonio@lcc.uma.es>
 */
@SuppressWarnings("serial")
public class RNSGAII<S extends Solution<?>> extends NSGAII<S> implements
    InteractiveAlgorithm<S,List<S>>, Measurable {

  private List<Double> interestPoint;
  private double epsilon;

  protected SimpleMeasureManager measureManager ;
  protected BasicMeasure<List<S>> solutionListMeasure ;
  protected CountingMeasure iterations ;
  protected DurationMeasure durationMeasure ;
 
  protected BasicMeasure<Double> hypervolumeValue ;
  protected BasicMeasure<Double> epsilonValue;
  protected BasicMeasure<Double> IGDValue;
    
  protected Front referenceFront ;

  /**
   * Constructor
   */
  public RNSGAII(Problem<S> problem, int maxIterations, int populationSize,
                 int matingPoolSize, int offspringPopulationSize,
                 CrossoverOperator<S> crossoverOperator, MutationOperator<S> mutationOperator,
                 SelectionOperator<List<S>, S> selectionOperator, SolutionListEvaluator<S> evaluator,
                 List<Double> interestPoint, double epsilon) {
    super(problem,maxIterations, populationSize,matingPoolSize, offspringPopulationSize, crossoverOperator,
            mutationOperator,selectionOperator, new DominanceComparator<S>(), evaluator);
    this.interestPoint = interestPoint;
    this.epsilon = epsilon;

    measureManager = new SimpleMeasureManager() ;
    measureManager.setPushMeasure("currentPopulation", solutionListMeasure);
    measureManager.setPushMeasure("currentEvaluation", iterations);
      
    measureManager.setPushMeasure("hypervolume", hypervolumeValue);
    measureManager.setPushMeasure("epsilon", epsilonValue);
    measureManager.setPushMeasure("IGD", IGDValue);

    referenceFront = new ArrayFront() ;

    initMeasures();
  }
    
  @Override
  public void updatePointOfInterest(List<Double> newReferencePoints){
    this.interestPoint = newReferencePoints;
  }
    
  @Override protected void initProgress() {
    iterations.reset(getMaxPopulationSize()) ;
  }

  @Override protected void updateProgress() {
    iterations.increment(getMaxPopulationSize());
    solutionListMeasure.push(getPopulation());
    epsilonValue.push(new Epsilon<S>(referenceFront).evaluate(getResult()));
    if (referenceFront.getNumberOfPoints() > 0) {
        hypervolumeValue.push(new PISAHypervolume<S>(referenceFront).evaluate(getResult()));
        IGDValue.push(new InvertedGenerationalDistance<S>(referenceFront).evaluate(getResult()));
    }  
      
  }

  @Override protected boolean isStoppingConditionReached() {
    return iterations.get() >= maxIterations;
  }

  @Override
  public void run() {
    durationMeasure.reset();
    durationMeasure.start();
    super.run();
    durationMeasure.stop();
  }

  /* Measures code */
  private void initMeasures() {
    durationMeasure = new DurationMeasure() ;
    iterations = new CountingMeasure(0) ;
    solutionListMeasure = new BasicMeasure<>() ;
    hypervolumeValue = new BasicMeasure<>();
    epsilonValue = new BasicMeasure<>();
    IGDValue = new BasicMeasure<>();

    measureManager = new SimpleMeasureManager() ;
    measureManager.setPullMeasure("currentExecutionTime", durationMeasure);
    measureManager.setPullMeasure("currentEvaluation", iterations);
    measureManager.setPullMeasure("hypervolume", hypervolumeValue);
    measureManager.setPullMeasure("epsilon", epsilonValue);
    measureManager.setPullMeasure("IGD", IGDValue);

    measureManager.setPushMeasure("currentPopulation", solutionListMeasure);
    measureManager.setPushMeasure("currentEvaluation", iterations);
    measureManager.setPushMeasure("hypervolume", hypervolumeValue);
    measureManager.setPushMeasure("epsilon", epsilonValue);
    measureManager.setPushMeasure("IGD", IGDValue);
  }

  @Override
  public MeasureManager getMeasureManager() {
    return measureManager ;
  }

  @Override protected List<S> replacement(List<S> population, List<S> offspringPopulation) {
    List<S> jointPopulation = new ArrayList<>();
    jointPopulation.addAll(population);
    jointPopulation.addAll(offspringPopulation);

    RankingAndPreferenceSelection<S> rankingAndCrowdingSelection ;
    rankingAndCrowdingSelection = new RankingAndPreferenceSelection<S>(getMaxPopulationSize(), interestPoint, epsilon) ;

    return rankingAndCrowdingSelection.execute(jointPopulation) ;
  }

  @Override public List<S> getResult() {
    return getNonDominatedSolutions(getPopulation());
  }

  @Override public String getName() {
    return "RNSGAII" ;
  }

  @Override public String getDescription() {
    return "Reference Point Based Nondominated Sorting Genetic Algorithm version II" ;
  }
}
