package mars.venus;

import mars.*;
import java.awt.event.*;
import javax.swing.*;

public class SettingsFunctionLocalLabelsAction extends GuiAction {

	public SettingsFunctionLocalLabelsAction(String name, Icon icon, String descrip,
			Integer mnemonic, KeyStroke accel, VenusUI gui) {
		super(name, icon, descrip, mnemonic, accel, gui);
	}

	public void actionPerformed(ActionEvent e) {
		Globals.getSettings().setFunctionLocalLabels(
				((JCheckBoxMenuItem) e.getSource()).isSelected());
	}

}