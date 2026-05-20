# ============================================
# Author: Виктория Комарова (Vikitoria007)
# License: MIT
# GitHub: https://github.com/Vikitoria007/noisy-channel-transmission
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

def compress_quantize(image_array, levels=4):
    step = 256 // levels
    compressed = (image_array // step) * step
    return compressed 

def frame_to_bits(frame):
    """превращаем кадр в список битов"""
    pixels = compress_quantize(frame, levels = 4)
    shape = pixels.shape

    bits = []
    for row in pixels:
        for pixel in row:
            for bit_char in format(pixel, '08b'):
                bits.append(int(bit_char))
    return bits, shape 

def bits_to_frame(bits, shape):
    """превращаем обратно в кадр"""
    expected_pixels = shape[0] * shape[1]
    bits = bits[:expected_pixels * 8]

    pixels = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        bit_str = ''.join(str(b) for b in byte_bits)
        pixel_value = int(bit_str, 2)
        pixels.append(pixel_value)

    img_array = np.array(pixels, dtype=np.uint8).reshape(shape)
    return img_array

def qpsk_modulate(bits):
    """QPSK (Quadrature Phase Shift Keying) модуляция: биты -> комплексные символы."""
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
    """OFDM (Orthogonal Frequency-Division Multiplexing) модуляция: символы -> временной сигнал."""
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

def plot_ber_curve(bits, qpsk_symbols, snr_range = (0, 25, 3)):
    snr_values = range(snr_range[0], snr_range[1], snr_range[2])
    ber_list = []
    print("\n измеряем BER для SNR")
    for snr_db in snr_values:
        ofdm_signal = ofdm_modulate(qpsk_symbols)
        rx_signal = add_awgn_noise(ofdm_signal, snr_db)
        rx_symbols = ofdm_demodulate(rx_signal)
        rx_symbols = rx_symbols[:len(qpsk_symbols)]
        bits_recovered = qpsk_demodulate(rx_symbols)
        bits_recovered = bits_recovered[:len(bits)]

        errors = np.sum(np.array(bits) != np.array(bits_recovered))
        ber = errors / len(bits) if len(bits) > 0 else 1
        ber_list.append(ber) 
        print(f"SNR = {snr_db} dB, BER = {ber:.6f}")

    plt.figure(figsize = (8, 5))
    plt.semilogy(snr_values, ber_list, 'bo-', linewidth = 2, markersize = 8)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER (Bit Error Rate)")
    plt.title("Зависимость BER от отношения сигнал/шум")
    plt.grid(True, which = "both", linestyle = "--", alpha = 0.7)
    plt.savefig("ber_curve.png", dpi = 150)
    print("\n График BER сохранен как ber_curve.png")

if __name__ == "__main__":
    print ("передаем MP4 с котенком-программистом")

    # показ исходного видео
    print ("показываем исходное цветное видео")
    cap_show = cv2.VideoCapture("kitten.mp4")
    while True:
        ret, frame = cap_show.read()
        if not ret:
            break
        cv2.imshow("Original Video (Color)", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break 
    cap_show.release()
    cv2.destroyAllWindows()
    print("показ исходного видео завершен")

    # загружаем кадры
    cap = cv2.VideoCapture("kitten.mp4")
    original_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64))
        original_frames.append(small)
    cap.release()

    print(f"Всего кадров: {len(original_frames)}")

    # BER 
    first_frame = original_frames[0]
    bits, shape = frame_to_bits(first_frame)
    print(f"Битов в кадре: {len(bits)}")

    # QPSK модуляция
    qpsk_symbols = qpsk_modulate(bits)
    print (f"QPSK-символов: {len(qpsk_symbols)}")

    plot_ber_curve(bits, qpsk_symbols, snr_range = (0, 25, 3))

    # передаем все кадры
    recovered_frames = []
    snr = 20

    for idx, frame in enumerate(original_frames):
        bits, shape = frame_to_bits(frame)
        qpsk_symbols = qpsk_modulate(bits) 
        ofdm_signal = ofdm_modulate(qpsk_symbols)
        rx_signal = add_awgn_noise(ofdm_signal, snr)
        rx_symbols = ofdm_demodulate(rx_signal)
        rx_symbols = rx_symbols[:len(qpsk_symbols)]
        bits_recovered = qpsk_demodulate(rx_symbols)
        bits_recovered = bits_recovered[:len(bits)]
        recovered_frame = bits_to_frame(bits_recovered, shape)
        recovered_frames.append(recovered_frame)

        if (idx + 1) % 5 == 0:
            print(f"Обработано кадров: {idx+1}/{len(original_frames)}")
        
    # созвездие qpsk
    fig1, axes = plt.subplots(1, 2, figsize = (10, 5))

    axes[0].imshow(original_frames[0], cmap = "gray") 
    axes[0].set_title("исходная кадр (котенок)")
    axes[0].axis("off")

    axes[1].imshow(recovered_frames[0], cmap = "gray")
    axes[1].set_title(f"после передачи по зашумленному каналу (SNR = {snr} dB)")

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
    print ("Восстановленное видео: recovered_kitten.avi")

    # покадровое сравнение
    print("генерируем покадровое сравнение")
    num_comparisons = min(5, len(original_frames))

    for i in range(num_comparisons):
        fig, axes = plt.subplots(1, 2, figsize = (8, 4))

        axes[0].imshow(original_frames[i], cmap="gray")
        axes[0].set_title(f"исходный кадр {i+1}")
        axes[0].axis("off")

        axes[1].imshow(recovered_frames[i], cmap="gray")
        axes[1].set_title(f"восстановленный кадр {i+1} (SNR = {snr} dB)")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(f"frame_comparison_{i+1}.png")
        plt.close()
    print(f"сохранено {num_comparisons} покадровых сравнений")
    
    print ("\n программма завершена")
    plt.show()
