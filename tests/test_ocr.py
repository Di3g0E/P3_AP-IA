import easyocr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_loader import load_images

# Carga el lector una sola vez (fuera de la función)
reader = easyocr.Reader(['es', 'en'])

def ocr_easy(img_rgb):
    results = reader.readtext(img_rgb)
    texto = "\n".join([r[1] for r in results])
    return texto

def corregir_errores_ocr(texto):
    """
    Corrige confusiones comunes entre letras y números en OCR.
    """
    reemplazos = {
        'O': '0',
        'o': '0',
        'I': '1',
        'l': '1',
        'Z': '2',
        'S': '5',
        'B': '8'
    }
    corregido = ""
    for c in texto:
        corregido += reemplazos.get(c, c)
    return corregido

import re

def extraer_numeros(texto):
    """
    Extrae todos los valores numéricos que parezcan importes.
    """
    texto = corregir_errores_ocr(texto)

    patron = r"""
        (?<!\w)                # no letra antes
        [€$]?\s*               # símbolo opcional
        \d{1,3}                # 1-3 dígitos
        (?:[.,\s]\d{3})*       # miles opcionales
        [.,]\d{2}              # decimales obligatorios
        (?!\w)                 # no letra después
    """

    numeros = re.findall(patron, texto, re.VERBOSE)
    return numeros


def filtrar_no_totales(texto, valores):
    texto_lower = texto.lower()

    palabras_prohibidas = [
        "cash", "paid", "change", "cambio", "efectivo",
        "devuelto", "pago", "pagado"
    ]

    valores_filtrados = []
    for v in valores:
        # buscamos la línea donde aparece el valor
        for linea in texto_lower.split("\n"):
            if v in linea:
                if not any(p in linea for p in palabras_prohibidas):
                    valores_filtrados.append(v)
                break

    return valores_filtrados


def seleccionar_total(valores):
    """
    Selecciona el valor monetario más alto como total.
    """
    def normalizar(v):
        v = v.replace("€", "").replace("$", "").strip()
        v = v.replace(",", ".")
        return float(v)

    if not valores:
        return None

    valores_float = [normalizar(v) for v in valores]
    total = max(valores_float)
    return total


def extraer_total_factura(texto_ocr):
    # 1. corregir errores
    # texto_corregido = corregir_errores_ocr(texto_ocr)

    # 2. extraer valores
    valores = extraer_numeros(texto_ocr)

    # 3. filtrar valores no válidos
    valores_filtrados = filtrar_no_totales(texto_ocr, valores)

    # 4. seleccionar total
    total = seleccionar_total(valores_filtrados)

    return total


# Ejemplo de uso
images = load_images(path="data/raw")
texto = ocr_easy(images[1])

print(texto)
