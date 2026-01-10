package mars.mips.instructions.syscalls;

import mars.util.*;
import mars.mips.hardware.*;
import mars.*;

/**
 * Service to display on the console float whose bits are stored in $f12
 */
@SuppressWarnings("deprecation")
public class SyscallPrintFloat extends AbstractSyscall {
	/**
	 * Build an instance of the Print Float syscall.  Default service number
	 * is 2 and name is "PrintFloat".
	 */
	public SyscallPrintFloat() {
		super(2, "PrintFloat");
	}

	/**
	* Performs syscall function to display float whose bits are stored in $f12
	*/
	public void simulate(ProgramStatement statement) throws ProcessingException {
		SystemIO.printString(new Float(Float.intBitsToFloat(
				Coprocessor1.getValue(12))).toString());
	}
}