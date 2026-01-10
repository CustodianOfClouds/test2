package mars.venus;

import mars.*;
import java.awt.event.*;
import javax.swing.*;

/**
 * Action class for the Settings menu item to control number base (10 or 16) of memory/register contents.
 */
@SuppressWarnings("deprecation")
public class SettingsValueDisplayBaseAction extends GuiAction {

	public SettingsValueDisplayBaseAction(String name, Icon icon, String descrip,
			Integer mnemonic, KeyStroke accel, VenusUI gui) {
		super(name, icon, descrip, mnemonic, accel, gui);
	}

	public void actionPerformed(ActionEvent e) {
		boolean isHex = ((JCheckBoxMenuItem) e.getSource()).isSelected();
		Globals.getGui().getMainPane().getExecutePane().getValueDisplayBaseChooser().setSelected(isHex);
		Globals.getSettings().setDisplayValuesInHex(isHex);
	}

}