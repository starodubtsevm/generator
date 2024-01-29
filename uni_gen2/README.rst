# trc3
Тестовая dsp модель универсального генератора.
Типы генерируемых сигналов:

- синусоидальный (SIN,COS);
- АМ (AM);
- ЧИМ (FSK);
- ФМ (PSK);
- 2ФРМ (QPSK).

Форматы выходных данных:

- numpy.array;
- wav файл;
- выход звуковой карты (реальное время).

Интерфейс:

class generator (object):
    
    def __init__(self, fc = 475, fm =8, Ampl = 100,
        fdev=13, fskspeed = 11, code, fs = 2000) 

        self.Ampl = Ampl        
        self.fc = fc
        self.fs = fs
    
    

    def AM(self):
        """генератор сигнала АМ"""
    
    def FSK(self):
        """генератор сигнала ЧИМ"""

    def PSK(self):
        """генератор сигнала ФИМ"""

    def QPSK(self):
        """генератор сигнала 2ФРМ"""

    def SetAmpl(self, A):
        self.Ampl = A

    def SetFcar(self,fc):
        self.fc = fc

    def SetFs(self,fs):
        self.fs = fs

Вызов:

generator1 = generator(fc, fm, Ampl, fs)

generator1.AM.Audio()
generator1.FSK.NumpyArray()
generator.PSK.Wav()


