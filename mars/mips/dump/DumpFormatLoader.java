package mars.mips.dump;

import mars.util.*;
import java.util.*;
import java.lang.reflect.*;

/****************************************************************************/
/* This class provides functionality to bring external memory dump format definitions
 * into MARS.  This is adapted from the ToolLoader class
 */
@SuppressWarnings({ "deprecation", "rawtypes", "unchecked" })
public class DumpFormatLoader {

	private static final String CLASS_PREFIX = "mars.mips.dump.";
	private static final String DUMP_DIRECTORY_PATH = "mars/mips/dump";
	private static final String CLASS_EXTENSION = "class";

	private static ArrayList formatList = null;

	/**
	 *  Dynamically loads dump formats into an ArrayList.  This method is adapted from
	 *  the loadGameControllers() method. 
	 * Also see the ToolLoader and SyscallLoader classes elsewhere in MARS.
	 */

	public ArrayList loadDumpFormats() {
		// The list will be populated only the first time this method is called.
		if (formatList == null) {
			formatList = new ArrayList();
			// grab all class files in the dump directory
			ArrayList<String> candidates = FilenameFinder.getFilenameList(this.getClass().getClassLoader(),
					DUMP_DIRECTORY_PATH, CLASS_EXTENSION);
			for (int i = 0; i < candidates.size(); i++) {
				String file = (String) candidates.get(i);
				try {
					// grab the class, make sure it implements DumpFormat, instantiate, add to list
					String formatClassName = CLASS_PREFIX + file.substring(0, file.indexOf(CLASS_EXTENSION) - 1);
					Class<?> clas = Class.forName(formatClassName);
					if (DumpFormat.class.isAssignableFrom(clas) &&
							!Modifier.isAbstract(clas.getModifiers()) &&
							!Modifier.isInterface(clas.getModifiers()))
						formatList.add(clas.newInstance());
				} catch (Exception e) {
					System.out.println("Error instantiating DumpFormat from file " + file + ": " + e);
				}
			}
		}
		return formatList;
	}

	public static DumpFormat findDumpFormatGivenCommandDescriptor(ArrayList formatList,
			String formatCommandDescriptor) {
		DumpFormat match = null;
		for (int i = 0; i < formatList.size(); i++) {
			if (((DumpFormat) formatList.get(i)).getCommandDescriptor().equals(formatCommandDescriptor)) {
				match = (DumpFormat) formatList.get(i);
				break;
			}
		}
		return match;
	}

}
