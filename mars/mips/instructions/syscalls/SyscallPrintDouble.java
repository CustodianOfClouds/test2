package mars.mips.instructions.syscalls;

import mars.util.*;
import mars.mips.hardware.*;
import mars.*;

/**
 * Service to display double whose bits are stored in $f12 & $f13 onto the console.
 * $f13 contains high order word of the double.
 */
@SuppressWarnings("deprecation")
public class SyscallPrintDouble extends AbstractSyscall {
	/**
	 * Build an instance of the Print Double syscall.  Default service number
	 * is 3 and name is "PrintDouble".
	 */
	public SyscallPrintDouble() {
		super(3, "PrintDouble");
	}

	/**
	* Performs syscall function to print double whose bits are stored in $f12 & $f13.
	*/
	public void simulate(ProgramStatement statement) throws ProcessingException {
		// Note: Higher numbered reg contains high order word so concat 13-12.
		SystemIO.printString(new Double(Double.longBitsToDouble(
				Binary.twoIntsToLong(Coprocessor1.getValue(13), Coprocessor1.getValue(12)))).toString());
	}
}