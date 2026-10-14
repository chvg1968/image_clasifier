# Clasificador de avisos publicitarios

Detecta avisos en fotos de calle, incluidas las panorámicas 360° de 12K. Dibuja la máscara de cada aviso y lo clasifica en **Valla**, **Fachada** o **Dudoso**.

## Cómo funciona

```
panorámica ─► mosaicos 2048 px ─► SAM 3 (texto → cajas + máscaras) ─► unión de duplicados
           ─► por cada objeto: recorte + contexto + ejemplos de referencia ─► Haiku / OpenAI
           ─► anotada.jpg + máscaras + recortes + resultado.json + resultados.csv
```

1. **Detección y segmentación**, de forma local en el Mac:
   - `sam3` (por defecto): detecta y segmenta desde texto con las frases `sign, store sign, billboard, advertisement, banner, poster`. Necesita `HF_TOKEN` y la licencia aceptada en Hugging Face.
   - `dino-sam2`: Grounding DINO + SAM 2.1. Es la alternativa sin token.
   - Las panorámicas (proporción 2:1) se procesan en mosaicos de 2048 px a resolución completa. Solo se recorre la franja entre el 18 % y el 70 % de la altura, para saltar el cielo cenital y el capó del vehículo.
2. **Clasificación** de cada objeto. El clasificador ve tres cosas:
   - los **ejemplos de referencia** de `assets/images_examples`, con su etiqueta correcta;
   - el **recorte** del objeto;
   - una **ventana de contexto** con el objeto enmarcado en rojo, para saber si está sobre un poste, una pared o un vehículo.

   Puedes elegir el clasificador:
   - `haiku`: Claude Haiku 5.5 con salida JSON. Devuelve la categoría, si el aviso es legible, el texto transcrito y una razón. Los ejemplos van en caché.
   - `openai`: OpenAI Decisions API (`gpt-6-luna`). Devuelve probabilidades de "es publicidad", de cada categoría y de "es legible".
   - `both` (recomendado): corre los dos. La categoría final es la de Haiku, salvo cuando Haiku dice Dudoso y OpenAI dice Fachada o Valla con probabilidad ≥ 0.8. En ese caso gana OpenAI (ver `combine` en `app/pipeline.py`).
3. **Unión de fragmentos de valla:** si SAM 3 detecta por separado partes de una misma valla (franja de precio, logo, borde), `merge_billboard_parts` en `app/segment.py` las une en un solo objeto con la máscara rellena.
4. **Filtro de ruido**, antes de clasificar: los objetos cuyo lado mayor mide menos de 60 px, o con score de detección menor que 0.5, se descartan sin llamar a la API. En una panorámica 12K casi todos son manchas imposibles de identificar. Quedan en `descartados` con el motivo "filtrado". Se ajusta con `--min-size` y `--min-score`.

## Ejemplos de referencia

La etiqueta de cada ejemplo sale del nombre del archivo (`app/examples.py`):

| Prefijo | Significado |
|---|---|
| `ej_valla_*` | Valla |
| `ej_fachada_*` | Fachada, incluidos los avisos parcialmente tapados |
| `ej_borde_totem_*` | Tótem sobre su propio poste: **Valla**, por decisión del equipo |
| `ej_no_dibujar_*` | No se marca: publicidad en vehículos y señales informativas |
| `ej_no_dudoso_*` | Aviso tapado en parte pero identificable: **no es Dudoso** |

Para agregar un ejemplo, pon la imagen con el aviso enmarcado en `assets/images_examples/` con uno de esos prefijos. Si quieres explicar el caso, agrega una nota en `NOTAS`, en `app/examples.py`.

## Uso

```bash
.venv/bin/python -m app assets/images --classifier both     # lote completo
.venv/bin/python -m app assets/images --skip-done           # continúa un lote interrumpido
.venv/bin/python -m app.reclassify output output_v2 --classifier both   # reclasifica sin volver a segmentar
.venv/bin/python -m app.report output_v2                    # página de revisión
.venv/bin/python -m app.ui                                  # interfaz web en http://127.0.0.1:7860
```

## Salidas

- `output/resultados.csv`: una fila por objeto detectado, con la categoría de cada clasificador.
- `output/revision.html`: recortes con las etiquetas de cada clasificador. Los desacuerdos aparecen primero.
- `output/<imagen>/`:
  - `anotada.jpg`: imagen anotada a la mitad de resolución.
  - `crops/`: recortes de cada objeto.
  - `masks/`: máscaras recortadas a cada aviso.
  - `resultado.json`: los campos `bbox_xyxy` y `poligono` están en píxeles de la imagen original. `mascara_origen_xy` indica dónde va el PNG de la máscara dentro de la imagen original.

## Reglas de la categoría final

- `NoPublicidad`: vehículos, señales de tránsito o informativas y objetos que no son avisos. Van a `descartados` y no se dibujan.
- `NoPublicidad` también incluye los objetos que no se pueden identificar y no muestran ninguna señal comercial.
- `Dudoso`: el objeto tiene alguna señal comercial pero no es claro si es publicidad, o es claramente un aviso que no se puede leer. Una Valla ilegible sigue siendo Valla, porque se reconoce por su estructura.
  - Con OpenAI también pasa a Dudoso cuando la confianza de la categoría es menor que 0.6.
- Los criterios completos están en `CRITERIOS`, dentro de `app/classify.py`.
- Se probó una "regla de oro" más estricta (Valla/Fachada solo si el modelo declara lectura clara, commit `8a29854`). Bajó la precisión de 79 % a 69 % en la muestra aleatoria de 150, porque Haiku marcaba como poco claros letreros que sí había transcrito. No se usa.

## Limitaciones conocidas

- La unión entre el borde izquierdo y el derecho de la panorámica (lo que queda detrás del vehículo) no se procesa en conjunto. Un aviso partido justo ahí puede salir en dos pedazos.
- Cada panorámica tarda unos 70 s en un Mac M4, casi todo en SAM 3.

## Etiquetado manual y precisión

```bash
.venv/bin/python -m app.label_page output_v2          # genera output_v2/etiquetar.html
.venv/bin/python -m app.evaluate output_v2 ~/Downloads/etiquetas_manuales.csv
.venv/bin/python -m app.label_page output_v4 --muestra 150 --excluir ~/Downloads/etiquetas_manuales.csv   # muestra aleatoria
```

Configuración recomendada para lotes nuevos: solo Haiku. La ventana de contexto va a 768 px por defecto (`CONTEXT_MAX_SIDE`).

```bash
.venv/bin/python -m app assets/images --classifier haiku
```

`etiquetar.html` muestra los objetos en los que Haiku y OpenAI no coinciden, mezclados con 30 acuerdos de control. Por defecto oculta las respuestas de los modelos. Las etiquetas se guardan en el navegador y "Descargar CSV" las exporta. Las etiquetas consolidadas están en `etiquetas/`. Como la numeración de objetos cambia entre reclasificaciones, `app.remap_labels` traslada las etiquetas a una carpeta nueva comparando cajas, y `app.eval_subset` compara modelos clasificando solo los objetos etiquetados. `app.evaluate` calcula cuánto acierta cada clasificador (haiku, openai y final) contra esas etiquetas.
