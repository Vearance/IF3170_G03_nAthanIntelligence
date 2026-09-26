from dataclasses import dataclass

"""
	Evaluator akan memeriksa sebuah state:
		1. Tidak menembus dinding truk
			x >= 0
			x + width <= truck.width

			(Berlaku juga untuk y dan z).
			
		2. Tidak bertabrakan dengan paket lain (tidak boleh ada dua box di satu titik koordinat posisi)
		
		3. Tidak melayang
		
		4. Paket di lantai truk valid.
			Kalau tidak di lantai, harus memiliki support dari paket lain.
			Support terjadi jika bagian atas paket lain tepat berada di bawah alas paket.
			
		5. Paket yang isFragile==True tidak boleh menjadi support
			
		6. Tidak melebihi kapasitas
			total weight paket di truk <= maxCapacity

	NB: Paket di luar truk tidak ikut pengecekan collision dan tidak dihitung dalam kapasitas.
"""

@dataclass
class Evaluate:
	pass