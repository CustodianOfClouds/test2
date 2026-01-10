package mars.mips.instructions.syscalls;

import mars.mips.hardware.*;
import mars.*;
import java.util.Random;

/**
 * Service to set seed for the underlying Java pseudorandom number generator. No values are returned.
 *
 */
@SuppressWarnings({ "deprecation", "unchecked" })
public class SyscallRandSeed extends AbstractSyscall {
	/**
	 * Build an instance of the syscall with its default service number and name.
	 */
	public SyscallRandSeed() {
		super(40, "RandSeed");
	}

	/**
	* Set the seed of the underlying Java pseudorandom number generator.
	*/
	public void simulate(ProgramStatement statement) throws ProcessingException {
		// Arguments: $a0 = index of pseudorandom number generator
		//   $a1 = seed for pseudorandom number generator.
		// Result: No values are returned. Sets the seed of the underlying Java pseudorandom number generator.

		Integer index = new Integer(RegisterFile.getValue(4));
		Random stream = (Random) RandomStreams.randomStreams.get(index);
		if (stream == null)
			RandomStreams.randomStreams.put(index, new Random(RegisterFile.getValue(5)));
		else
			stream.setSeed(RegisterFile.getValue(5));
	}

}
