'''
T1DelaySweep
=======================================================================
An NVAveragerProgram class acquires data for T1 decay measurement, i.e. 
a mw pluse sequence pi - delay - readout where the pi pulse is cycled between
on and off acquire contrasted measurements
'''


from .nvaverageprogram import NVAveragerProgram
from .nvqicksweep import NVQickSweep

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os 

class T1DelaySweep(NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates and executes a sequence used
    to measure the T1 Decay

    Parameters
    -------------------------------------------------------------------
    soccfg
        instance of qick.QickConfig class
    cfg
        instance of qickdawg.NVConfiguration class with attributes:
        .adc_channel (required)
            int channel which is reading data 0 or 1
        
        .mw_channel (required)
            qick channel that provides microwave excitation
            0 or 1 for RFSoC4x2
            0 to 6 for ZCU111 or ZCU216
        .mw_nqz (required)
            nyquist zone for microwave generator (1 or 2)
        .mw_gain (required)
            gain of micrwave channel, in register values, from 0 to 2**15-1
        
        .pre_init (required)
            boolian value that indicates whether to pre-pulse the laser to initialize
            the spin state

        .relax_delay_treg (required)
            int that indicates how long to delay between on/off cycles and reps
            in register units
        .readout_length_treg (required)
            int time for which the adc accumulates data
            the limit is 1020 points for the FPGA buffer
        .laser_readout_offset_treg (required)

        .laser_gate_pmod(required)
            int PMOD channel used to trigger laser source
            0 to 4

    Methods
    -------
    initialize
        method that generates the assembly code that setups the adcs &  mw generators,
        and performs other one-off setps
    body
        method that generates the assembly code that exectues in the middle of each sweep
        and rep
    plot_sequence
        generates a plot labeled with self.cfg attributes or the required inputs
    time_per_rep
        returns the approximatetime for one rep to complete
    total_time
        returns the approximate total time for the entire program to complete
    '''
    required_cfg = [
        "adc_channel",
        "readout_integration_treg",
        "mw_channel",
        "mw_nqz",
        "mw_pi2_treg",
        "mw_gain",
        "scaling_mode",
        "delay_start_treg",
        "delay_end_treg",
        "nsweep_points",
        "pre_init",
        "laser_gate_pmod",
        "laser_on_treg",
        "relax_delay_treg",
        "reps",
        "readout_reference_start_treg",
        "laser_readout_offset_treg",
        "mw_readout_delay_treg"]

    def initialize(self):
        '''
        Method that generates the assembly code that is sets up adcs and sources. 
        For T1DecaySweep this:
        1. configures the adc to acquire points for self.cfg.readout_integration_t#. 
        2. configures the microwave channel 
        3. configures the sweep parameters
        4. initiailzes the spin state with a laser pulse
        '''

        self.check_cfg()
        self.declare_readout(ch=0,
                             freq=0,
                             length=self.cfg.readout_integration_treg,
                             sel="input")

        ## Setup Microwave Channel

        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)
        
        self.default_pulse_registers(
            ch=self.cfg.mw_channel,
            style='const',
            freq=self.cfg.mw_freg,
            length=self.cfg.mw_pi2_treg,
            gain=self.cfg.mw_gain)

        self.set_pulse_registers(ch=self.cfg.mw_channel, phase=0)

        # Add loops
        self.delay_register = self.new_gen_reg(self.cfg.mw_channel,
                                               name='delay', 
                                               init_val=self.cfg.delay_start_treg)

        if self.cfg.scaling_mode == 'exponential':
            self.add_sweep(NVQickSweep(
                self, 
                self.delay_register,
                self.cfg.delay_start_treg, 
                self.cfg.delay_end_treg,
                expts=self.cfg.nsweep_points,
                scaling_mode=self.cfg.scaling_mode,
                scaling_factor=self.cfg.scaling_factor))

        elif self.cfg.scaling_mode == 'linear':
            self.add_sweep(NVQickSweep(
                self, 
                self.delay_register,
                self.cfg.delay_start_treg, 
                self.cfg.delay_end_treg,
                self.cfg.nsweep_points))

        self.synci(400)  # give processor some time to self.cfgure pulses

        if self.cfg.pre_init:
            self.trigger(
                pins=[self.cfg.laser_gate_pmod],
                width=self.cfg.laser_on_treg,
                adc_trig_offset=0
            )           
            self.sync_all(self.cfg.laser_on_treg + self.cfg.relax_delay_treg)

    def body(self):
        '''
        Method that generates the assembly code that is looped over or repeated. 
        For T1DecaySweep this peforms four measurements at a time and does two pulse
        sequences differing as wot whether the microwave is on or off on the first pulse.
        The sequences is:
        1. Pulse mw for pi (just delay for second sequence)
        2. delay by variable delay time
        3. Perform readout
        4. Loop over delay times
        5. Loop over reps
        6. Loop over rounds
        '''

        ## First pulse sequence
        ## pi(x) - delay - readout
        # pi2(x)
        self.pulse(ch=self.cfg.mw_channel, t=0)
        self.pulse(ch=self.cfg.mw_channel, t=self.cfg.mw_pi2_treg)
        self.synci(self.cfg.mw_pi2_treg * 2)
        # delay
        self.sync(self.delay_register.page, self.delay_register.addr)
        # readout
        self.sync_all(self.cfg.mw_readout_delay_treg)        
        self.ttl_readout()
        
        ## Second pulse sequence
        ## pi(x) off - delay - readout
        ## Second pulse sequence
        ## Nothing - delay - readout
        # pi(x) - off
        self.synci(self.cfg.mw_pi2_treg * 2)
        # delay
        self.sync(self.delay_register.page, self.delay_register.addr) 
        # readout
        self.sync_all(self.cfg.mw_readout_delay_treg)
        self.ttl_readout()
    
    def acquire(self, raw_data=False, *arg, **kwarg):

        data = super().acquire(reads_per_rep=4, *arg, **kwarg)

        if raw_data is False:
            data = self.analyze_pulse_sequence_results(data)
            data.contrast = data.contrast * -1

        return data

    def plot_sequence(cfg=None):
        '''
        Function that plots the pulse sequence generated by this program
        
        Parameters
        ----------
        cfg: `.NVConfiguration` or None(default None)
            If None, this plots the squence with configuration labels
            If a `.NVConfiguration` object is supplied, the configuraiton value are added to the plot
        '''
        graphics_folder = os.path.join(os.path.dirname(__file__), '../../graphics')
        image_path = os.path.join(graphics_folder, 'T1.png')


        if cfg is None:
            plt.figure(figsize=(12, 12))
            plt.axis('off')
            plt.imshow(mpimg.imread(image_path))
            plt.text(500, 700, "config.reps", fontsize=14)
      
            plt.text(305, 335, "delay", fontsize=10)
            plt.text(400, 385, "  config.readout_reference_start", fontsize=10)
            plt.text(250, 335, "pi", fontsize=10)

            plt.text(260, 465, "config.laser_readout_offset", fontsize=10)
            plt.text(390, 340, "config.readout_integration", fontsize=10)
            plt.text(650, 340, "config.readout_integration", fontsize=10)
            plt.text(850, 340, "config.relax_delay", fontsize=10)
            plt.text(400, 430, "config.laser_on", fontsize=10)
            plt.text(220, 605, "Sweep delay from config.delay_start to config.delay_end in config.nsweep_points \n                            with scaling given by config.scaling_mode", fontsize=12)
            plt.title("             T1 Pulse Sequence", fontsize=20)
        else:
            plt.figure(figsize=(12, 12))
            plt.axis('off')
            plt.imshow(mpimg.imread(image_path))
            plt.text(450, 700, "Repeat {} times".format(cfg.reps), fontsize=14)
            plt.text(305, 335, "delay", fontsize=10)
            plt.text(400, 385, "  config.readout_reference_start", fontsize=10)
            plt.text(250, 335, "pi", fontsize=10)
            plt.text(240, 465, "laser_readout_offset = {} treg".format(cfg.laser_readout_offset_treg), fontsize=10)
            plt.text(390, 337, "readout_integration = {} us".format(str(cfg.readout_integration_tus)[:4]), fontsize=10)
            plt.text(650, 357, "readout_integration \n = {} us".format(str(cfg.readout_integration_tus)[:4]), fontsize=10)
            plt.text(850, 357, "relax_delay \n = {} us".format(str(cfg.relax_delay_tus)[:4]), fontsize=10)
            plt.text(400, 430, "laser_on = {} us".format(cfg.laser_on_tus), fontsize=12)
            plt.text(325, 605, "    Sweep delay from {} us to {} us \n                in {} {} steps".format(int(cfg.delay_start_tns), int(cfg.delay_end_tns), cfg.nsweep_points, cfg.scaling_mode), fontsize=12)
            plt.title("               T1 Pulse Sequence", fontsize=20)
            
    def save(self, data, folder_path = None, folder_name = None, separate_dates = True, additional_configs = []):
        '''
        Method that saves a file containing the raw data given as well as two graphs,
        one of the odmr contrast and another showing the reference and signal
        
        Parameters
        ----------
        data
            The raw data to be saved. Should be the output of the analyze_results function
        folder_path
            Location that the folder contaning the saved files should be (if separated dates = True this will be the location
            of the folder of folders)
        separated_dates
            Boolean value which if true will create a subfolder for the day the measurment was saved. All measurments saved
            that day will be put in the corresponding folder
        folder_name
            The name of the file the data and plots will be saved in. If None it will be the data and time
        additional_configs
            A list of 2 item arrays, with the first item containing the name of the config and the second item containing the 
            value. These will be saved alongside the values in the config item used to generate the data in a text file
        
        '''
        
        folder_path = self.init_save(self.cfg, data, folder_path, folder_name, separate_dates)
        
        # plot the signal and reference and save it
        
        p0 = [0.6, 5e3, 0]

        plt.plot(data.sweep_tus, data.contrast)
        #plt.plot(d.sweep_tus, qd.exponential_decay(d.sweep_tus, *p0))

        param, _ = curve_fit(qd.exponential_decay, data.sweep_tus, data.contrast, p0)
        plt.plot(d.sweep_tus, qd.exponential_decay(data.sweep_tus, *param))

        plt.xscale('log')
        # plt.xlim(1, )
        print('T1 is approximatel {:0f} ms'.format(param[1]/1e3))
        plt.title('T1 Relaxation')
        plt.ylabel('Contrast (%)')
        plt.xlabel('Delay (us)')
        
        
        plt.savefig(folder_path + '/Rabi_Contrast_and_Reference.png')
        plt.clf()
        
        # save the measurment configurations in a textfile
        
        with open(folder_path + '/Config.txt', "w") as file:
            file.write('Readout time (us): ' + str(self.cfg.readout_integration_tus) + '\n')
            file.write('Laser on time (us): ' + str(self.cfg.laser_on_tus) + '\n')
            file.write('Laser readout offset (us): ' + str(self.cfg.laser_readout_offset_tus) + '\n')
            file.write('Readout reference start time (us): ' + str(self.cfg.readout_reference_start_tus) + '\n')
            file.write('Repetitions: ' + str(self.cfg.reps) + '\n')
            file.write('Laser power  (arb): ' + str(self.cfg.laser_power) + '\n')
            file.write('MW gain (arb): ' + str(self.cfg.mw_gain) + '\n')
            file.write('Pi/2 pulse length (ns): ' + str(self.cfg.mw_pi2_tns) + '\n')
            file.write('MW frequency (MHz): ' + str(self.cfg.mw_fMHz) + '\n')
            for config in additional_configs:
                file.write(str(config[0]) + ': ' + str(config[1]) + '\n')

