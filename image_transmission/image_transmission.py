from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

def image_to_bits(image_path):
    """Загружает картинку и превращает в список битов (0 и 1)."""
    img = Image.open(image_path).convert('L')
    img.thumbnail((64,64))
    pixels = np.array(img)
    shape = pixels.shape

    bits = []
    for row in pixels:
        for pixel in row:
            for bit_char in format(pixel, '08b'):
                bits.append(int(bit_char))
    return bits, shape 

def bits_to_image(bits, shape):
    expected_pixels = shape[0] * shape[1]
    bits = bits[:expected_pixels * 8]

    pixels = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        bit_str = ''.join(str(b) for b in byte_bits)
        pixel_value = int(bit_str, 2)
        pixels.append(pixel_value)

    img_array = np.array(pixels, dtype=np.uint8).reshape(shape)
    return Image.fromarray(img_array, mode='L')

def qpsk_modulate(bits):
    """QPSK модуляция: биты -> комплексные символы."""
    symbols = []
    for i in range(0, len(bits) - 1, 2):
        b1 = bits[i]
        b2 = bits[i + 1]

        if b1 == 0 and b2 == 0:
            symbols.append(1 + 1j)
        elif b1 == 0 and b2 == 1:
            symbols.append(-1 + 1j)
        elif b1 == 1 and b2 == 0:
            symbols.append(1 - 1j)
        else:
            symbols.append(-1 - 1j)

    return np.array(symbols)

def qpsk_demodulate(symbols):
    """QPSK демодуляция: комплексные символы -> биты."""
    bits = []
    for s in symbols:
        if s.real > 0 and s.imag > 0:
            bits.extend([0, 0])
        elif s.real < 0 and s.imag > 0:
            bits.extend([0, 1])
        elif s.real > 0 and s.imag < 0:
            bits.extend([1, 0])
        else:
            bits.extend([1, 1])
    return bits

def ofdm_modulate(symbols, num_subcarries = 256, cp_length = 32):
    """OFDM модуляция: символы -> временной сигнал."""
    data_per_symbol = num_subcarries - 1
    total_symbols = len(symbols)
    num_ofdm_symbols = (total_symbols + data_per_symbol - 1) // data_per_symbol

    tx_signal = []
    for i in range(num_ofdm_symbols):
        start = i * data_per_symbol 
        end = min(start + data_per_symbol, total_symbols)
        block = symbols[start:end]

        # Размещаем символы на поднесущих
        freq_domain = np.zeros(num_subcarries, dtype = complex)
        freq_domain[1:len(block) + 1] = block 

        # IFFT и циклический префикс
        time_domain = np.fft.ifft(freq_domain)
        cp = time_domain[-cp_length:]

        tx_signal.extend(np.concatenate([cp, time_domain]))

    return np.array(tx_signal)

def ofdm_demodulate(rx_signal, num_subcarries=256, cp_length=32):
    """OFDM демодуляция: сигнал -> QPSK-символы."""
    data_per_symbol = num_subcarries - 1
    symbol_len = num_subcarries + cp_length 
    num_ofdm_symbols = len(rx_signal) // symbol_len

    rx_symbols = []
    for i in range(num_ofdm_symbols):
        start = i * symbol_len
        ofdm_symbol = rx_signal[start:start + symbol_len]

        # Убираем циклический префикс и делаем FFT
        time_domain = ofdm_symbol[cp_length:]
        freq_domain = np.fft.fft(time_domain)
        rx_symbols.extend(freq_domain[1:num_subcarries])
    return np.array(rx_symbols)

def add_awgn_noise(signal, snr_dp):
    """Добавляет белый гауссовский шум к сигналу."""
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_dp / 10)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise 

if __name__ == "__main__":
    print ("используется логотип МТУСИ: my_photo.png")

    # Загрузка изображения в биты
    bits, shape = image_to_bits("my_photo.png")
    print (f"Битов: {len(bits)}")

    # QPSK модуляция
    qpsk_symbols = qpsk_modulate(bits)
    print (f"QPSK-символов: {len(qpsk_symbols)}")

    # OFDM модуляция
    ofdm_signal = ofdm_modulate(qpsk_symbols)
    print (f"OFDM отсчетов: {len(ofdm_signal)}")

    # Добавление шума (канал связи)
    snr = 20
    rx_signal = add_awgn_noise(ofdm_signal, snr)
    print (f"добавлен шум SNR: {snr} дб")

    # OFDM демодуляция
    rx_symbols = ofdm_demodulate(rx_signal)
    rx_symbols = rx_symbols[:len(qpsk_symbols)]

    # QPSK демодуляция
    bits_recovered = qpsk_demodulate(rx_symbols)  
    bits_recovered = bits_recovered[:len(bits)]

    # Проверка качества передачи
    if np.array_equal(bits, bits_recovered):
        print ("биты совпадают")
    else:
        errors = np.sum(np.array(bits) != np.array(bits_recovered))
        print (f"ошибок: {errors} из {len(bits)}, BER = {errors / len(bits):.6f}, биты не совпали")

    # Восстановление изображения
    recovered_img = bits_to_image(bits_recovered, shape)
    recovered_img.save("recovered_image.png")

    # Визуализация: сравнение картинок
    print("генерация графиков")

    fig1, axes = plt.subplots(1, 2, figsize = (10, 5))
    original = Image.open("my_photo.png")

    axes[0].imshow(original, cmap = "gray")
    axes[0].set_title("исходная картинка")
    axes[0].axis("off")

    axes[1].imshow(recovered_img, cmap = "gray")
    axes[1].set_title(f"после 5G (SNR = {snr} dB)")

    plt.tight_layout()
    plt.savefig("1_comparison.png")
    print("Done. 1_comparison.png")

    # Визуализация: созвездие QPSK
    fig2, axes = plt.subplots(1, 2, figsize = (10, 5))

    axes[0].scatter(qpsk_symbols.real, qpsk_symbols.imag, s = 3, alpha = 0.5)
    axes[0].set_title("передатчик qpsk")
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-2, 2)
    axes[0].grid(True)
    axes[0].set_xlabel("Re")
    axes[0].set_ylabel("Im")

    axes[1].scatter(rx_symbols.real, rx_symbols.imag, s = 3, alpha = 0.5, c = "red")
    axes[1].set_title(f"приемник (SNR = {snr} dB)")
    axes[1].set_xlim(-2, 2)
    axes[1].set_ylim(-2, 2)
    axes[1].grid(True)
    axes[1].set_xlabel("Re")

    plt.tight_layout()
    plt.savefig("2constellation.png")
    print ("Done. 2constellation.png")
    
    print ("графики сохранены")
    print ("Восстановленная картинка сохранена как recovered_image.png")
    plt.show()
