package mars.venus;

import java.awt.event.*;
import javax.swing.*;

/**
* Action  for the File -> Open menu item
*/
public class FileOpenAction extends GuiAction {

	public FileOpenAction(String name, Icon icon, String descrip,
			Integer mnemonic, KeyStroke accel, VenusUI gui) {
		super(name, icon, descrip, mnemonic, accel, gui);
	}

	/**
	* Launch a file chooser for name of file to open
	*
	* @param e component triggering this call
	*/
	public void actionPerformed(ActionEvent e) {
		mainUI.editor.open();
	}

}
