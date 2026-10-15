"""Preprocesado y vuelta a la rejilla original propios de bifusion, sobre mambalite3d.data sin modificarlo.

- restore: vuelta lineal de las probabilidades a la rejilla de la etiqueta original.
- labels: remuestreo de la etiqueta a la rejilla de trabajo ganglio por ganglio, y etiquetas suaves.
- pet_scale: deteccion de un PET que no esta en SUV y factor para corregirlo.
"""
