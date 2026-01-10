package mars.mips.instructions.syscalls;

import mars.*;
import mars.util.*;
import mars.mips.hardware.*;

/**
 * Service to read a character from input console into $a0.
 *
 */

public class SyscallReadChar extends AbstractSyscall {
	/**
	 * Build an instance of the Read Char syscall.  Default service number
	 * is 12 and name is "ReadChar".
	 */
	public SyscallReadChar() {
		super(12, "ReadChar");
	}

	/**
	* Performs syscall function to read a character from input console into $a0
	*/
	public void simulate(ProgramStatement statement) throws ProcessingException {
		Globals.inputSyscallLock.lock();
		try {
			int value = SystemIO.readChar(this.getNumber());

			if (value == -1)
				throw new ProcessingException();

			RegisterFile.updateRegister(2, value);
		} finally {
			Globals.inputSyscallLock.unlock();
		}
	}

}