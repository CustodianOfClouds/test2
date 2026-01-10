package mars.venus;

import mars.*;
import java.awt.event.*;
import javax.swing.*;

/**
* Action  for the Help -> About menu item
*/
public class HelpAboutAction extends GuiAction {
	public HelpAboutAction(String name, Icon icon, String descrip,
			Integer mnemonic, KeyStroke accel, VenusUI gui) {
		super(name, icon, descrip, mnemonic, accel, gui);
	}

	public void actionPerformed(ActionEvent e) {
		JOptionPane.showMessageDialog(mainUI,
				"MARS " + Globals.version + "\n" +
						"MARS is the Mips Assembler and Runtime Simulator.\n\n" +
						"Totally a legit kind of language ngl lmao\n",
				"About Mars",
				JOptionPane.INFORMATION_MESSAGE,
				new ImageIcon("images/RedMars50.gif"));
	}
}