package mars.mips.instructions;

import mars.*;

/**
 * Interface to represent the method for simulating the execution of a specific MIPS basic
 * instruction.  It will be implemented by the anonymous class created in the last
 * argument to the BasicInstruction constructor.
 *
 * @author CC
 * @version the big 26
 *
 */

public interface SimulationCode {

	/**
	 * Method to simulate the execution of a specific MIPS basic instruction.
	 *
	 * @param statement A ProgramStatement representing the MIPS instruction to simulate.
	 * @throws ProcessingException This is a run-time exception generated during simulation.
	 **/

	public void simulate(ProgramStatement statement) throws ProcessingException;
}
