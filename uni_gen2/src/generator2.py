import numpy as np
from sys import exit
from time import time
from os import system
from pylab import figure
from scipy.fft import fft
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.io.wavfile import write
from scipy.fftpack import fft, fftshift

class generator:
    """класс - универсальный генератор сигналов"""

    count = 0
    fc = 525
    fs = 8000
    simTime = 20
    Time = np.arange(0, simTime, 1 / fs)
    Phase = np.zeros(len(Time))
    Ampl = np.ones(len(Time))

    Byte2 = 0x2C
    Byte1 = 0x1F

    fsk_code = 0x2C
    fsk_bit_rate = 13

    qpsk_bit_rate = 11

    f_am_mod = 0
    alsn_code = "G"

    modulation = "none"

    mu = 0
    sigma = 0.1

    def __init__(self, plotsig="True"):

        self.plotsig = plotsig
        self.__class__.count += 1

    def go(self):
        """старт генератора"""
        print("\033[32m" + " " + self.modulation + " сигнал")
        if self.modulation == "AM":
            return self.AM()
        elif self.modulation == "FSK":
            return self.FSK()
        elif self.modulation == "PSK":
            return self.PSK()
        elif self.modulation == "ALSEN":
            return self.ALSEN()
        elif self.modulation == "Noise":
            return self.Noise()
        elif self.modulation == "ALSN":
            return self.ALSN()
        else:
            print("\033[31mОшибка конфигурации генератора")
        exit(1)

    def setAmpl(self, A):
        self.Ampl = self.Ampl * A

    def setSigma(self, sigma):
        self.sigma = sigma

    def setFcar(self, fc):
        self.fc = fc

    def setFs(self, fs):
        self.__class__.fs = fs
        self.__class__.Time = np.arange(0, self.simTime, 1 / self.fs)

    def setSimTime(self, simTime):
        self.simTime = simTime

    def plotEnable(self, flag):
        self.plotsig = flag

    def setCode(self, code):
        self.code = code

    def setAlsnCode(self, code):
        self.alsn_code = code

    def setModulation(self, mod):
        self.modulation = mod

    def setAMmodFreq(self, freq):
        self.f_am_mod = freq

    def setAlsenCodes(self, code1, code2):
        self.Byte1 = code1
        self.Byte2 = code2

    # --modulators-----------------------------------------------

    def AM(self):
        print("\033[32mгенерируем АМ сигнал")

        return self.Signal(self.Ampl, self.Phase)

    def FSK(self):
        print("\033[32mгенерируем ЧИМ сигнал")
        counter = 0
        data_sig = []
        Phase = []
        h = 1.5
        Tb = 1 / self.fsk_bit_rate
        limit = self.fs // self.fsk_bit_rate
        bit_num = 0
        bauer_code = np.unpackbits(np.array([self.code], dtype=np.uint8))
        while counter < len(self.Time):
            bit = bauer_code[bit_num]
            counter += 1
            if counter % limit == 0:
                bit_num = (bit_num + 1) % 8
            if bit == 0:
                b = -1
            else:
                b = +1
            data_sig.append(b)
        b_integrated = lfilter([1.0], [1.0, -1.0], data_sig) / self.fs
        Phase = np.pi * h / Tb * b_integrated  # theta
        # s = self.Ampl*np.cos(2 * np.pi * self.fc * self.Time + theta)
        return self.Signal(self.Ampl, Phase)

    def ALSEN(self):
        print("\033[32mгенерируем 2ФРМ сигнал (АЛСЕН)")
        self.fmod = 0
        count_bit = 8
        diBit = 0
        phase = 0
        d_phase = 0
        Phase = []
        byte1 = self.Byte1
        byte2 = self.Byte2
        d_phase = [0, np.pi / 2, 3 / 2 * np.pi, np.pi]

        for n, _ in enumerate(self.Time):
            if n % int((1 / self.qpsk_bit_rate) / (1 / self.fs)) == 0:
                if count_bit == 0:
                    count_bit = 8
                    byte1 = self.Byte1
                    byte2 = self.Byte2
                diBit = ((byte1 & 0x80) >> 6) + ((byte2 & 0x80) >> 7)
                byte1 = byte1 << 1
                byte2 = byte2 << 1
                phase = phase + d_phase[diBit]
                if phase > 2 * np.pi:
                    phase -= 2 * np.pi
                count_bit = count_bit - 1
            Phase.append(phase)
        Phase = np.array(Phase)
        # f = self.Ampl * np.sin(2 * np.pi * self.fc * self.Time + Phase)
        return self.Signal(self.Ampl, Phase)

    def ALSN(self):

        print("\033[32mгенерируем сигнал АЛСН")
        alsn = []
        count_ticks = 0
        self.f_am_mod = 0

        green_code = {
            "pulse1": 0.35,
            "pause1": 0.12,
            "pulse2": 0.22,
            "pause2": 0.12,
            "pulse3": 0.22,
            "pause3": 0.57,
        }
        yellow_code = {
            "pulse1": 0.38,
            "pause1": 0.12, 
            "pulse2": 0.38,
            "pause2": 0.72
        }
        redyellow_code = {
            "pulse1": 0.23,
            "pause1": 0.57
        }

        if self.alsn_code == "G":
            elements = green_code
        elif self.alsn_code == "Y":
            elements = yellow_code
        elif self.alsn_code == "RY":
            elements = redyellow_code

        t = 0
        while t <= (len(self.Time)):
            for element in elements:
                count_ticks = 0
                while elements.get(element) >= count_ticks:
                    if element[:-1] == "pause":
                        alsn.append(0)
                    elif element[:-1] == "pulse":
                        alsn.append(1)
                    count_ticks += 1.0 / self.fs
                    t += 1
        alsn = alsn[0 : len(self.Time)]
        alsn = np.array(alsn)
        alsn = alsn * self.Ampl
        return self.Signal(alsn, self.Phase)

    def PSK(self):

        print("\033[32mгенерируем ФИМ сигнал")
        print("\033[32mпока не реализовано")
        return 0

    # ---------------------------------------------------------------------

    def Noise(self):

        # Add AWGN noise to the transmitted Signal
        print("\033[32mгенерируем шум")
        noise = self.mu + self.sigma * np.random.normal(1, 1, len(self.Time))
        return noise

    def Signal(self, ampl, Phase):

        signal = (
            ampl
            * np.cos(2 * np.pi * self.fc * self.Time + Phase)
            * (1 + np.cos(2 * np.pi * self.f_am_mod * self.Time))
        )
        return signal

    def make_wave_scaled(self, data, name="none"):
        print("\033[32mсоздаем wav файл")
        rate = self.fs
        scaled = np.int16(data / np.max(np.abs(data)) * 32767)
        if self.modulation == "AM":
            name = "Сигнал АМ " + str(self.fc) + " " + str(self.f_am_mod) + ".wav"
        elif self.modulation == "FSK":
            name = "Сигнал ЧМ " + str(self.fc) + " " + str(self.code) + ".wav"
        elif self.modulation == "QPSK":
            name = "Сигнал АЛСЕН " + str(self.fc) + " " + str(self.Byte2) + ".wav"
        elif self.modulation == "ALSN":
            name = "Сигнал АЛСН " + str(self.fc) + " " + str(self.alsn_code) + ".wav"
        elif self.modulation == "mix":
            name = name
        else:
            pass
        write(name, rate, scaled)
        return 0

    def make_plot(self, data, N):

        print("\033[32mсоздаем графики")
        figure(N)
        ax1 = plt.subplot(211)
        ax1.plot(self.Time, data)
        if self.modulation == "AM":
            ax1.set_title("Сигнал " + str(self.fc) + " / " + str(self.f_am_mod) + " Гц")
        if self.modulation == "FSK":
            ax1.set_title("Сигнал " + str(self.fc) + " / " + str(self.code))
        if self.modulation == "mix":
            ax1.set_title("Смесь сигналов")
        if self.modulation == "ALSN":
            ax1.set_title("АЛСН " + str(self.fc) + " / " + str(self.alsn_code))
        if self.modulation == "ALSEN":
            ax1.set_title("АЛСЕН " + str(self.fc) + " / " + str(self.Byte1) + " Гц")

        ax1.grid(True)

        ax2 = plt.subplot(212)
        ax2.set_title("Спектр сигнала ")
        ax2.grid(True)
        return 0

    def make_spectr(self, data, N):
        print("\033[32mвычисляем спектр")
        figure(N)
        plt.grid(True)
        plt.ylim(0, 60)
        plt.xlim(0, 1000)
        plt.magnitude_spectrum(data, Fs=self.fs, scale="dB")
        plt.ylabel("Уровень (dB)")
        plt.xlabel("Частота (Hz)")
        plt.tight_layout()
        return 0

    def calc_power(self, data):
        print("\033[32mвычисляем мощность")
        Power = (norm(data) ** 2) / len(data)
        return Power


# ------------------------------------------------------
if __name__ == "__main__":
    system("clear")
# -noise------------------------------------------------
start = time()
gennoise = generator()
gennoise.modulation = "Noise"
gennoise.setFs(8000)
gennoise.setSigma(4)
noise = gennoise.go()
gennoise.make_wave_scaled(noise)
# gennoise.make_plot(noise,1)

p = gennoise.calc_power(noise)
print("\033[33mМощность шума (временная область {:0.4f}".format(p))

# -сигнал 1---------------------------------------------
gen1 = generator()
gen1.modulation = "ALSEN"
gen1.setAlsenCodes(0x4a, 0xe0)
gen1.setFs(8000)
gen1.setFcar(174)
gen1.setAmpl(384)
# gen1.setCode(0xab)
# gen1.setAMmodFreq(12)
#gen1.setAlsnCode("RY")
signal = gen1.go()

gen1.make_wave_scaled(signal)
gen1.make_plot(signal, 2)
# gen1.make_spectr(signal, 2)

p = gen1.calc_power(signal)
print("\033[33mМощность сигнала (временная область {:0.4f}".format(p))

# -сигнал 2---------------------------------------------
gen2 = generator()
gen2.modulation = "FSK"
gen2.setFs(8000)
gen2.setFcar(725)
gen2.setAmpl(256)
gen2.setCode(0xe0)
#gen2.setAMmodFreq(12)
signal2 = gen2.go()

gen2.make_wave_scaled(signal2)
gen2.make_plot(signal2, 3)
# gen2.make_spectr(signal2, 3)

p = gen2.calc_power(signal2)
print("\033[33mМощность сигнала (временная область {:0.4f}".format(p))

# -сумма сигнал 1 + noise---------------------
mix = [a + b + c for a, b, c in zip(noise, signal2, signal)]
gen2.setModulation("mix")
name = "Смесь сигналов  " + str(gen1.fc) + " " + str(gen2.fc) + ".wav"
gen2.make_wave_scaled(mix, name)
# gen2.make_plot(mix, 4)
# gen2.make_spectr(mix, 4)

finish = time() - start
print(finish)
# -------------------------------------------------------
plt.show()
