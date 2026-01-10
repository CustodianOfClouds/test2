package mars.venus;

import mars.*;
import java.awt.event.*;
import javax.swing.*;

/**
 * Action class for the Settings menu item to control automatic assemble of file upon opening.
 */
public class SettingsClearRunIOOnAssembleAction extends GuiAction {

	public SettingsClearRunIOOnAssembleAction(String name, Icon icon, String descrip,
			Integer mnemonic, KeyStroke accel, VenusUI gui) {
		super(name, icon, descrip, mnemonic, accel, gui);
	}

	public void actionPerformed(ActionEvent e) {
		Globals.getSettings().setClearRunIOOnAssemble(
				((JCheckBoxMenuItem) e.getSource()).isSelected());
	}

}