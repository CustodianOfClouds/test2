package mars.mips.instructions.syscalls;

import mars.util.*;
import mars.mips.hardware.*;
import mars.*;

/**
 * Service to display integer stored in $a0 on the console.
 *
 */
@SuppressWarnings("deprecation")
public class SyscallPrintInt extends AbstractSyscall {
	/**
	 * Build an instance of the Print Integer syscall.  Default service number
	 * is 1 and name is "PrintInt".
	 */
	public SyscallPrintInt() {
		super(1, "PrintInt");
	}

	/**
	* Performs syscall function to print on the console the integer stored in $a0.
	*/
	public void simulate(ProgramStatement statement) throws ProcessingException {
		SystemIO.printString(
				new Integer(RegisterFile.getValue(4)).toString());
	}
}