package mars.mips.hardware;

/**
 * Represents attempt to access double precision register using an odd
 * (e.g. $f1, $f23) register name.
 *
 * @author CC
 * @version the big 26
 **/

public class InvalidRegisterAccessException extends Exception {
	/**
	 * Constructor for IllegalRegisterException.
	 *
	 **/
	public InvalidRegisterAccessException() {
		// do nothing
	}

}
