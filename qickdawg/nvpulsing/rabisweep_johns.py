'''
RabiSweep
=======================================================================
An NVAveragerProgram class that generates and executes a measurement which takes
four measurements while sweeping the microwave output length to generate a Rabi
oscillation dataset which can be used to determine the pi and pi/2 pulse lenghts
for future measurements
'''

from qick.averager_program import QickSweep
from .nvqicksweep import NVQickSweep
from .nvaverageprogram_johns import NVAveragerProgram_johns
from ..util import ItemAttribute
import qickdawg as qd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import pickle
import os 
import serial

class RabiSweep_johns(NVAveragerProgram_johns):
    '''
    An NVAveragerProgram class that generates and executes a sequence used
    to determine the pi (pi/2) pulse lenghts for your experimetanl configuration

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
    returns
        an instances of LockinODMR class with assembly language compiled


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
    required_cfg = ["adc_channel",
                    "readout_integration_treg",
                    "mw_channel",
                    "mw_nqz",
                    "mw_gain",
                    "mw_freg",
                    "mw_start_treg",
                    "mw_end_treg",
                    "nsweep_points",
                    "pre_init",
                    "laser_gate_pmod",
                    "relax_delay_treg",
                    "reps",
                    "mw_delta_treg",
                    ]

    def initialize(self):
        '''
        Method that generates the assembly code that is sets up adcs and sources. 
        For RabiSweep this:
        configures the adc to acquire points for self.cfg.readout_integration_t#. 
        configures the microwave channel 
        configures the sweep parameters
        initiailzes the spin state with a laser pulse
        '''
        self.check_cfg()
        self.declare_readout(ch=self.cfg.adc_channel,
                             freq=0,
                             length=self.cfg.readout_integration_treg,
                             sel="input")

        # configure pulse defaults and initial parameters for microwave
        self.declare_gen(
            ch=self.cfg.mw_channel,
            nqz=self.cfg.mw_nqz)        

        self.default_pulse_registers(
            ch=self.cfg.mw_channel,
            style='const',
            freq=self.cfg.mw_freg,
            gain=self.cfg.mw_gain,
            phase=0)

        self.set_pulse_registers(ch=self.cfg.mw_channel,
                                 length=self.cfg.mw_start_treg)

        # configure the sweep
        self.mw_length_register = self.new_gen_reg(self.cfg.mw_channel, 
                                                   name='length', 
                                                   init_val=self.cfg.mw_start_treg)

        self.add_sweep(NVQickSweep(self, 
                                   reg=self.mw_length_register, 
                                   start=self.cfg.mw_start_treg, 
                                   stop=self.cfg.mw_end_treg, 
                                   expts=self.cfg.nsweep_points,
                                   label='length',
                                   mw_channel=self.cfg.mw_channel))

        self.synci(400)  # give processor some time to configure pulses

        if self.cfg.pre_init:

            self.trigger(
                pins=[self.cfg.laser_gate_pmod],
                width=self.cfg.laser_initialize_treg, 
                adc_trig_offset=self.cfg.adc_trigger_offset_treg)
            self.sync_all(self.cfg.laser_initialize_treg)

        self.wait_all()
        self.sync_all(self.cfg.relax_delay_treg)

    def body(self):
        '''
        Method that generates the assembly code that is looped over or repeated. 
        For RabiSweep this peforms four measurements at a time and does two pulse sequences:
        1. Microwave pulse followed by readout and reference emasurement
        2. No micrwave pulse followed by readout and reference 
        '''
        
        t = 0
                
        self.trigger(
            adcs=[],
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.laser_initialize_treg,
            adc_trig_offset=0,
            t=t)
        
        t += self.cfg.laser_initialize_treg
                        
        self.pulse(ch=self.cfg.mw_channel, t=t)

        self.sync(self.mw_length_register.page, self.mw_length_register.addr) 
        
        t += self.cfg.mw_delay_treg
        
        self.trigger(
            adcs=[self.cfg.adc_channel],
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.readout_integration_treg,
            adc_trig_offset=self.cfg.adc_trig_offset_treg,
            t=t)
        
        t += self.cfg.readout_integration_treg
        
        self.trigger(
            adcs=[],
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.laser_initialize_treg,
            adc_trig_offset=0,
            t=t)
                        
        t += self.cfg.mw_delay_treg
        
        t += self.cfg.mw_delay_treg + self.cfg.mw_end_treg
        
        self.trigger(
            adcs=[self.cfg.adc_channel],
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.readout_integration_treg,
            adc_trig_offset=self.cfg.adc_trig_offset_treg,
            t=t)

        self.sync_all(self.cfg.relax_delay_treg)
        self.wait_all()
        

    def acquire(self, raw_data=False, *arg, **kwarg):

        data = super().acquire(reads_per_rep=2, *arg, **kwarg)
        
        if raw_data is False:
            data = self.analyze_results(data)

        return data
    
    def analyze_results(self, data):
        """
        Method that takes in a 1D array of data points from self.acquire() and analyzes the
        results based on the number of reps, rounds, and frequency points

        Parameters
        ----------
        data
            (1D np.array) data returned from self.acquire()

        returns
            (qickdawg.ItemAttribute instance) with attributes
            .sweep_treg (len(nsweep_points) np array, reg units) - sweep lengths
            .sweep_tus (len(nsweep_points) np array, us units) - sweep lengths
            .signal1, .signal2 (nfrequency np.array, adc units)
                - average adc signal for microwave on, off
            .reference1, .reference2 (nfrequency np.array, adc units)
                - average referenceadc signal for microwave on, off
            .contrast1, contrast2 (nfrequency np.array, adc units))
                - (.signal1(2) minus .refrence1(2))/reference1(2)*100
            .contrast (nfrequency np.array, adc units)
                - .contrast1 - .contrast2
        """

        data = np.reshape(data, self.data_shape)
        data = data / self.cfg.readout_integration_treg

        if len(self.data_shape) == 2:
            signal = data[:, 0]
            reference = data[:, 1]
        elif len(self.data_shape) == 3:
            signal = data[:, :, 0]
            reference = data[:, :, 1]
        elif len(self.data_shape) == 4:
            signal = data[:, :, :, 0]
            reference = data[:, :, :, 1]

        for _ in range(len(signal.shape) - 1):
            signal = np.mean(signal, axis=0)
            reference = np.mean(reference, axis=0)

        d = ItemAttribute()
        d.signal = signal
        d.reference = reference
        d.contrast = (signal - reference) / reference * 100

        d.sweep_treg = self.qick_sweeps[0].get_sweep_pts()
        d.sweep_tus = self.qick_sweeps[0].get_sweep_pts() * self.cycles2us(1)
        
        return d
    
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
        image_path = os.path.join(graphics_folder, 'RABI.png')


        if cfg is None:
            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.imshow(mpimg.imread(image_path))
            plt.text(455, 510, "config.reps", fontsize=14)
            plt.text(350, 440, "config.laser_on", fontsize=14)
            plt.text(195, 580, " Sweep pi/2 pulse time linearly from config.mw_start to config.mw_end in config.mw_delta sized steps", fontsize=12)
            plt.text(265, 355, "config.readout_integration", fontsize=14)
            plt.text(527, 355, " config.readout_integration", fontsize=14)
            plt.text(190, 368, " pi/2\npulse", fontsize=14)
            plt.text(735, 355, "config.relax_delay", fontsize=14)
            plt.text(220, 407, "config.laser_readout_offset", fontsize=14)
            plt.text(430, 407, "config.readout_reference_start", fontsize=14)
            plt.title("           Rabi Oscillation Pulse Sequence", fontsize=20)

        else:
            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.imshow(mpimg.imread(image_path))
            plt.text(420, 510, "Repeat {} times".format(cfg.reps), fontsize=14)
            plt.text(350, 440, "laser_on_tus = {} us".format(str(cfg.laser_on_tus)[:4]), fontsize=14)
            plt.text(195, 580, " Sweep pi/2 pulse time linearly from {} time register to {} time register in steps of {} time register".format(int(cfg.mw_start_treg), int(cfg.mw_end_treg), str(cfg.mw_delta_treg)[:4]), fontsize=12)
            plt.text(265, 370, "readout_integration  \n       = {} ns".format(int(cfg.readout_integration_tns)), fontsize=14)
            plt.text(527, 370, "readout_integration  \n      = {} ns".format(int(cfg.readout_integration_tns)), fontsize=14)
            plt.text(190, 368, " pi/2\npulse", fontsize=14)
            plt.text(735, 370, "relax_delay \n = {} ns".format(int(cfg.relax_delay_tns)), fontsize=14)
            plt.text(235, 407, "laser_offset = {} ns".format(int(cfg.laser_readout_offset_tns)), fontsize=14)
            plt.text(430, 407, "readout_reference_start = {} us".format(int(cfg.readout_reference_start_tus)), fontsize=14)
            plt.title("           Rabi Oscillation Pulse Sequence", fontsize=20)
            
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
        
        # plot the signal and save it
        
        plt.plot(data.sweep_tus, data.signal, label='signal')

        plt.legend()
        plt.title('Rabi Oscillations')
        plt.ylabel('PL Intensity (arb)')
        plt.xlabel('pulse time (us)')
        
        plt.savefig(folder_path + '/Rabi_Sweep_Signal_and_Reference.png')
        plt.clf()
        
        # plot the contrast and save it
        
        plt.plot(data.sweep_tus, data.contrast, label='signal')

        plt.legend()
        plt.title('Rabi Oscillations')
        plt.ylabel('Contrast (%)')
        plt.xlabel('pulse time (us)')
        
        plt.savefig(folder_path + '/Rabi_Contrast_and_Reference.png')
        plt.clf()
        
        # save the measurment configurations in a textfile
        
        with open(folder_path + '/Config.txt', "w") as file:
            file.write('Readout time (us): ' + str(self.cfg.readout_integration_tus) + '\n')
            file.write('Relax time (us): ' + str(self.cfg.relax_delay_tus) + '\n')
            file.write('Mw delay time (us): ' + str(self.cfg.mw_delay_tus) + '\n')
            file.write('Laser intialize time (us): ' + str(self.cfg.laser_initialize_tus) + '\n')
            file.write('Repetitions: ' + str(self.cfg.reps) + '\n')
            file.write('Laser power: ' + str(self.cfg.laser_power) + '\n')
            file.write('MW gain: ' + str(self.cfg.mw_gain) + '\n')
            file.write('MW start time (us): ' + str(self.cfg.mw_start_tus) + '\n')
            file.write('MW end time (us): ' + str(self.cfg.mw_end_tus) + '\n')
            file.write('Number of sweep times sampled: ' + str(self.cfg.nsweep_points) + '\n')
            for config in additional_configs:
                file.write(str(config[0]) + ': ' + str(config[1]) + '\n')

