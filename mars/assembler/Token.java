package mars.assembler;

import mars.*;

/**
 * Represents one token in the input MIPS program.  Each Token carries, along with its
 * type and value, the position (line, column) in which its source appears in the MIPS program.
 *
 * @author CC
 * @version the big 26
 **/

public class Token {

	private TokenTypes type;
	private String value;
	private MIPSprogram sourceMIPSprogram;
	private int sourceLine, sourcePos;
	// original program and line will differ from the above if token was defined in an included file
	private MIPSprogram originalMIPSprogram;
	private int originalSourceLine;

	/**
	 * Constructor for Token class.
	 *
	 * @param type The token type that this token has. (e.g. REGISTER_NAME)
	 * @param value The source value for this token (e.g. $t3)
	 * @param sourceMIPSprogram The MIPSprogram object containing this token
	 * @param line The line number in source program in which this token appears.
	 * @param start The starting position in that line number of this token's source value.
	 * @see TokenTypes
	 **/

	public Token(TokenTypes type, String value, MIPSprogram sourceMIPSprogram, int line, int start) {
		this.type = type;
		this.value = value;
		this.sourceMIPSprogram = sourceMIPSprogram;
		this.sourceLine = line;
		this.sourcePos = start;
		this.originalMIPSprogram = sourceMIPSprogram;
		this.originalSourceLine = line;
	}

	/**
	 * Set original program and line number for this token.
	 * Line number or both may change during pre-assembly as a result
	 * of the ".include" directive, and we need to keep the original
	 * for later reference (error messages, text segment display).
	 *
	 * @param origProgram MIPS program containing this token.
	 * @param origSourceLine Line within that program of this token.
	 **/
	public void setOriginal(MIPSprogram origProgram, int origSourceLine) {
		this.originalMIPSprogram = origProgram;
		this.originalSourceLine = origSourceLine;
	}

	/**
	 * Produces original program containing this token.
	 *
	 * @return MIPSprogram of origin for this token.
	 **/
	public MIPSprogram getOriginalProgram() {
		return this.originalMIPSprogram;
	}

	/**
	 * Produces original line number of this token. It could change as result
	 * of ".include"
	 *
	 * @return original line number of this token.
	 **/
	public int getOriginalSourceLine() {
		return this.originalSourceLine;
	}

	/**
	 * Produces token type of this token.
	 *
	 * @return TokenType of this token.
	 **/
	public TokenTypes getType() {
		return type;
	}

	/**
	 * Set or modify token type.  Generally used to note that
	 * an identifier that matches an instruction name is
	 * actually being used as a label.
	 *
	 * @param type new TokenTypes for this token.
	 **/
	public void setType(TokenTypes type) {
		this.type = type;
	}

	/**
	 * Produces source code of this token.
	 *
	 * @return String containing source code of this token.
	 **/

	public String getValue() {
		return value;
	}

	/**
	 * Get a String representing the token.  This method is
	 * equivalent to getValue().
	 *
	 * @return     String version of the token.
	 */

	public String toString() {
		return value;
	}

	/**
	 * Produces MIPSprogram object associated with this token.
	 *
	 * @return  MIPSprogram object associated with this token.
	 **/

	public MIPSprogram getSourceMIPSprogram() {
		return sourceMIPSprogram;
	}

	/**
	 * Produces line number of MIPS program of this token.
	 *
	 * @return line number in source program of this token.
	 **/

	public int getSourceLine() {
		return sourceLine;
	}

	/**
	* Produces position within source line of this token.
	*
	* @return first character position within source program line of this token.
	**/

	public int getStartPos() {
		return sourcePos;
	}

}
