package mars.venus;

import mars.*;
import java.awt.event.*;
import javax.swing.*;

/**
 * Action class for the Settings menu item to control whether or not
 * assembler warnings are considered errors.  If so, a program generating
 * warnings but not errors will not assemble.
 */
@SuppressWarnings("deprecation")
public class SettingsStartAtMainAction extends GuiAction {

	public SettingsStartAtMainAction(String name, Icon icon, String descrip,
			Integer mnemonic, KeyStroke accel, VenusUI gui) {
		super(name, icon, descrip, mnemonic, accel, gui);
	}

	public void actionPerformed(ActionEvent e) {
		Globals.getSettings().setStartAtMain(((JCheckBoxMenuItem) e.getSource()).isSelected());
	}

}