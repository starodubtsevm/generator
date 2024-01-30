# trc3
Модель универсального генератора сигналов для использовани при dsp 
моделировании.
Типы генерируемых сигналов:

- синусоида (немодулированная частота);
- АМ (AM);
- ЧИМ (FSK);
- ФМ (PSK);
- 2ФРМ (QPSK).

Формирует специальные виды сигналов, применяемые на РЖД

- АЛСН  (setModulation(ALSN);
- АЛСЕН (setModulation(ALSEN);
- КРЛ   (setModulation(FSK).

Полученные wav файлы могут быть протестированы прибором типа ПКРЦ-М,
или аналогичным.

Форматы выходных данных:

- numpy.array;
- wav файл.

Интерфейс:

class generator:
    """класс - универсальный генератор сигналов"""

    def __init__(self, plotsig="True"):

        self.plotsig = plotsig
        self.__class__.count += 1s
    
Методы:    

    def go(self):
        """старт генератора"""
      
    def setAmpl(self, A):
        """установка амплитуды выходного сигнала, по умолчанию 1"""

    def setSigma(self, sigma):
        """установка значения sigma, для генератора шума, по умолчанию 0"""

    def setFcar(self, fc):
        """установка частоты генератора, по умолчанию 525 Гц"""

    def setFs(self, fs):
        """установка частоты дискретизации, по умолчанию 8000 Гц"""

    def setSimTime(self, simTime):
        """установка времени моделирования, по умолчанию 20 сек"""

    def plotEnable(self, flag):
        """установка флага вывода графиков на экран,по умолчанию True"""

    def setFskCode(self, code):
        """установка кода модуляции ЧИМ генератора, по умолчанию 0x2с"""

    def setAlsnCode(self, code):
        """установка кода сигнала АЛСН, по умолчанию G (код З)"""

    def setModulation(self, mod):
        """установка вида модуляции, по умолчанию none"""   

    def setAMmodFreq(self, freq):
        """установка частоты АМ модуляции, по умолчанию 0"""

    def setAlsenCodes(self, code1, code2):
        """установка кодов модуляции АЛСЕН генератора, по умолчанию 0x1f,0x2с"""
    
    def AM(self):
        """ формирователь АМ сигнала"""

    def FSK(self):
        """формирователь ЧИМ сигнала """

    def ALSEN(self):
        """формирователь сигнала АЛСЕН (QPSK) """

    def ALSN(self):
        """формирователь сигнала АЛСН"""

    def PSK(self):
        """формирователь фазоманипулированного сигнала"""

    def Noise(self):
        """формирователь шумоподобного сигнала"""

    def Signal(self, ampl, Phase):
        """формирователь синусоидального сигнала"""

    def make_wave_scaled(self, data, name="none"):
        """формирователь wav файла (нормированный сигнал)"""

    def make_plot(self, data, N):
        """функция построения графика сигнала и спектра сигнала)"""

    def make_spectr(self, data, N):
        """функция расчета спектра сигнала)"""

    def calc_power(self, data):
        """функция вычисления мощности сигнала"""

Примеры:

Генератор шума

gennoise = generator()
gennoise.modulation = "Noise"
gennoise.setFs(8000)
gennoise.setSigma(4) # амплитуда шума
noise = gennoise.go() # старт генератора
gennoise.make_wave_scaled(noise) # сгенерировать wav файл
# gennoise.make_plot(noise,1) # построить графики

p = gennoise.calc_power(noise) # расчет мощности сигнала
print("\033[33mМощность шума (временная область {:0.4f}".format(p))

Генератр КРЛ

gen2 = generator()
gen2.modulation = "FSK"  # тип сигнала
gen2.setFs(8000)  # частота дискретизации
gen2.setFcar(725) # частота несущей
gen2.setAmpl(256) # амплитуда сигнала
gen2.setFskCode(0xe0) # код для модуляции несущей
signal2 = gen2.go() # старт генератора

gen2.make_wave_scaled(signal2) # сгенерировать wav файл
gen2.make_plot(signal2, 3) # построить графики
# gen2.make_spectr(signal2, 3) # вычисление спектра
 
p = gen2.calc_power(signal2) # расчет мощности сигнала
print("\033[33mМощность сигнала (временная область {:0.4f}".format(p)

Формировние смеси из нескольких сигналов (например - два предыдущих)

# -сумма сигнал 1 + noise---------------------
mix = [a + b + c for a, b, c in zip(noise, signal2)]
gen2.setModulation("mix")
name = "Смесь сигналов  " + str(gen1.fc) + " " + str(gen2.fc) + ".wav"
gen2.make_wave_scaled(mix, name)
# gen2.make_plot(mix, 4)
# gen2.make_spectr(mix, 4)

