'''
LockinODMR
=======================================================================
An NVAveragerProgram class that generates and executes ODMR measurements by
measuring photoluminescnece intensity (PL) as a function of microwave frequency
while taking the difference between PL for microwave drive on or off
'''


from qick.averager_program import QickSweep
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

class LockinODMR_johns(NVAveragerProgram_johns):
    '''
    An NVAveragerProgram class that generates and executes ODMR measurements by
    measuring photoluminescnece intensity (PL) as a function of microwave frequency
    taking the difference between PL for microwave drive on or off

    Parameters
    -------------------------------------------------------------------
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
        .mw_start_freg(required)
            frequency generated by microwave channel in register units
        .mw_end_freg (required)
            frequency generated by micrewave channel at the end of the sweep
        .nsweep_points (required)
            number of points between mw_start_fMHz and mw_end_fMHz
        .mw_gain (required)
            gain of micrwave channel, in register values, from 0 to 2**15-1

        .pre_init (required)
            boolian value that indicates whether to pre-pulse the laser to initialize
            the spin state

        .relax_delay_treg (required)
            int that indicates how long to delay between on/off cycles and reps
            in register units
        .readout_integration_treg (required)
            int time for which PL intensity is averaged in register units

        .laser_gate_pmod(required)
            int PMOD channel used to trigger laser source
            0 to 4

        .reps(required)
            times the pulse sequence is repeated

        .reads_per_rep(required)
            times adcs are triggered per rep
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
    analyze_results
        takes the output of self.acquire() and returns odmr_contrast and other results
    plot_sequence
        generates a plot labeled with self.cfg attributes or the required inputs
    time_per_rep
        returns the approximatetime for one rep to complete
    total_time
        returns the approximate total time for the entire program to complete

    '''

    required_cfg = [
        "readout_integration_treg",
        "adc_channel",
        "laser_gate_pmod",
        "relax_delay_treg",
        "mw_channel",
        "mw_nqz",
        "mw_gain",
        "nsweep_points",
        "pre_init",
        "reps",
        "laser_power"
    
    ]

    def initialize(self):
        """
        Method that generates the assembly code that initializes the pulse sequence.
        For LockinODMR this sets up the adc to integrate for self.cfg.readout_intregration_t#,
        setups hte microave channel to run fo the same amount of time, and setups the a
        qickdawg.NVQickSweep to sweep over frequencies
        """
        self.check_cfg()
        if self.cfg.mw_gain > 30000:
            assert 0, 'Microwave gain exceeds maximum value'
        self.declare_readout(ch=self.cfg.adc_channel,
                             freq=0,
                             length=self.cfg.readout_integration_treg,
                             sel="input")


        # Get registers for mw
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        # Setup pulse defaults microwave
        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            style='const',
            freq=self.cfg.mw_start_freg,
            gain=self.cfg.mw_gain,
            length=self.cfg.readout_integration_treg,
            phase=0,
            stdysel='zero',
            mode='oneshot'
        )
        
        ## Get frequency register and convert frequency values to integers
        self.mw_frequency_register = self.get_gen_reg(self.cfg.mw_channel, "freq")

        self.add_sweep(QickSweep(self,
                                 self.mw_frequency_register,
                                 self.cfg.mw_start_fMHz,
                                 self.cfg.mw_end_fMHz,
                                 self.cfg.nsweep_points))

        self.synci(400)  # give processor some time to self.cfgure pulses

        if self.cfg.pre_init:
            self.pulse(ch=self.cfg.mw_channel)
            self.trigger(
                pins=[self.cfg.laser_gate_pmod],
                width=self.cfg.readout_integration_treg,
                adc_trig_offset=0)
            self.sync_all(self.cfg.readout_integration_treg + self.cfg.relax_delay_treg)

    def body(self):
        '''
        Method that generates the assembly code that is looped over or repeated.
        For LockinODMR this has two acquisitions
        The first acquisition has the microwave channel on for .cfg.readout_integration_t# and
            averages the adc values over this time
        The second acquisition has the microwave channel off for .cfg.readout_integration_t# and
            averages the adc values over this time
        '''

        self.trigger(
            adcs=[],
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.readout_integration_treg + self.cfg.adc_trigger_offset_treg,
            adc_trig_offset=0,
            t=0)
        
        self.trigger(
            adcs=[self.cfg.adc_channel],
            pins=[],
            width=self.cfg.readout_integration_treg,
            adc_trig_offset=0,
            t=self.cfg.adc_trigger_offset_treg)
        
        self.pulse(ch=self.cfg.mw_channel, t=self.cfg.adc_trigger_offset_treg)

        self.trigger(
            adcs=[],
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.readout_integration_treg + self.cfg.adc_trigger_offset_treg,
            adc_trig_offset=0,
            t=self.cfg.readout_integration_treg + self.cfg.relax_delay_treg + self.cfg.adc_trigger_offset_treg)
        
        self.trigger(
            adcs=[self.cfg.adc_channel],
            pins=[],
            width=self.cfg.readout_integration_treg,
            adc_trig_offset=0,
            t=self.cfg.readout_integration_treg + self.cfg.relax_delay_treg + 2*self.cfg.adc_trigger_offset_treg)

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
            .frequencies (len(frequencies) np array, GHz units) - frequencies swept over
            .signal (nfrequency np.array, adc units) - average adc signal for microwave on
            .reference (nfrequency np.array, adc units) - average adc signal for mwicrowave off
            .odmr (nfrequency np.array, adc units)) - .signal minus .refrence
            .odmr_contrast (nfrequency np.array, % units) - (.signal - .reference)/.reference *100
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

        odmr = (signal - reference)
        odmr_contrast = (signal - reference) / reference * 100

        for _ in range(len(odmr.shape) - 1):
            odmr = np.mean(odmr, axis=0)
            signal = np.mean(signal, axis=0)
            reference = np.mean(reference, axis=0)
            odmr_contrast = np.mean(odmr_contrast, axis=0)

        d = ItemAttribute()
        d.odmr = odmr
        d.signal = signal
        d.reference = reference
        d.odmr_contrast = odmr_contrast

        d.frequencies = self.qick_sweeps[0].get_sweep_pts()

        return d

    def time_per_rep(self):
        """
        Method that returns the approximate time per rep, excluding any overhead

        returns
            (float) time for one rep in seconds

        """
        t = self.cfg.readout_integration_tus * 2
        t += self.cfg.relax_delay_tus * 2
        t *= self.cfg.nsweep_points / 1e6

        return t

    def total_time(self):
        """
        Method that returns the approximate time for full qickdawg program to run

        returns
            (float) time for full qickdawg program in seconds
        """
        return self.time_per_rep() * self.cfg.reps

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
        image_path = os.path.join(graphics_folder, 'ODMR.png')
        
        if cfg is None:
            plt.figure(figsize=(10, 10))
            plt.axis('off')
            plt.imshow(mpimg.imread(image_path))
            plt.text(295, 340, "    config.reps", fontsize=16)
            plt.text(200, 275, "config.readout_integration_t#", fontsize=14)
            plt.text(520, 275, "config.relax_delay_t#", fontsize=14)
            plt.text(145, 430, "Sweep linearly from config.mw_start_f# to config.mw_end_f# \n                   in steps of config.mw_delta_f#", fontsize=14)
            plt.title("      ODMR Pulse Sequence", fontsize=20)
        else:
            plt.figure(figsize=(10, 10))
            plt.axis('off')
            plt.imshow(mpimg.imread(image_path))
            plt.text(295, 340, "Repeat {} times".format(cfg.reps), fontsize=16)
            plt.text(200, 275, "readout_integration = {} us".format(int(cfg.readout_integration_tus)), fontsize=14)
            plt.text(520, 290, "relax_delay \n = {} us".format(str(cfg.relax_delay_tus)[:4]), fontsize=14)
            plt.text(130, 400, "Sweep linearly from {} MHz to {} MHz in steps of {} MHz".format(int(cfg.mw_start_fMHz), int(cfg.mw_end_fMHz), str(cfg.mw_delta_fMHz)[:4]), fontsize=14)
            plt.title("      ODMR Pulse Sequence", fontsize=20)
            
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
        
        plt.plot(data.frequencies, data.signal, label='signal')
        plt.plot(data.frequencies, data.reference, label='reference')
        plt.title('ODMR Spectrum')
        plt.ylabel('fluorescence (arb)')
        plt.xlabel('frequency (MHz)')
        plt.legend()
        
        plt.savefig(folder_path + '/ODMR_Signal_and_Reference.png')
        plt.clf()
        
        # plot the odmr contrast and save it
        
        plt.plot(data.frequencies, -data.odmr_contrast)
        plt.title('ODMR Spectrum')
        plt.ylabel('contrast (%)')
        plt.xlabel('frequency (MHz)')
        
        plt.savefig(folder_path + '/ODMR_Contrast.png')
        plt.clf()
        
        # save the measurment configurations in a textfile
        
        with open(folder_path + '/Config.txt', "w") as file:
            file.write('Readout time (us): ' + str(self.cfg.readout_integration_tus) + '\n')
            file.write('Relax time (us): ' + str(self.cfg.relax_delay_tus) + '\n')
            file.write('Repetitions: ' + str(self.cfg.reps) + '\n')
            file.write('Laser power: ' + str(self.cfg.laser_power) + '\n')
            file.write('MW gain: ' + str(self.cfg.mw_gain) + '\n')
            file.write('MW start frequency: ' + str(self.cfg.mw_start_fMHz) + '\n')
            file.write('MW end frequency: ' + str(self.cfg.mw_end_fMHz) + '\n')
            file.write('Number of frequencies sampled: ' + str(self.cfg.nsweep_points) + '\n')
            for config in additional_configs:
                file.write(str(config[0]) + ': ' + str(config[1]) + '\n')
