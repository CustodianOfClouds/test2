package mars.mips.instructions.syscalls;

import mars.util.*;
import mars.mips.hardware.*;
import mars.*;

/**
 * Service to display string stored starting at address in $a0 onto the console.
 */
@SuppressWarnings("deprecation")
public class SyscallPrintString extends AbstractSyscall {
	/**
	 * Build an instance of the Print String syscall.  Default service number
	 * is 4 and name is "PrintString".
	 */
	public SyscallPrintString() {
		super(4, "PrintString");
	}

	/**
	* Performs syscall function to print string stored starting at address in $a0.
	*/
	public void simulate(ProgramStatement statement) throws ProcessingException {
		int byteAddress = RegisterFile.getValue(4);

		try {
			if (Memory.inDataSegment(byteAddress)) {
				// common case of printing a string from the data segment
				String str = Globals.memory.getAsciizFromDataSegment(byteAddress);
				SystemIO.printString(str);
			} else {
				// fall back to slow path
				char ch = (char) Globals.memory.getByte(byteAddress);
				// won't stop until NULL byte reached!
				while (ch != 0) {
					SystemIO.printString(new Character(ch).toString());
					byteAddress++;
					ch = (char) Globals.memory.getByte(byteAddress);
				}
			}
		} catch (AddressErrorException e) {
			throw new ProcessingException(statement, e);
		}
	}
}