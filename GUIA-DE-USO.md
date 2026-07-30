# Guía de uso — DD-Copilot

Esta guía explica, paso a paso y sin dar nada por sabido, qué hace cada
pieza del proyecto, cómo instalarlo y cómo usarlo. La versión en inglés
(mismo contenido) está en [`USER-GUIDE.md`](USER-GUIDE.md).

## Antes de nada: 3 conceptos clave

- **LLM (Large Language Model)**: un modelo de IA como Claude, capaz de
  leer texto y responder con texto (resumir, extraer datos, razonar sobre
  ello). Cada vez que le "preguntamos" algo, eso se llama una **llamada**.
- **RAG (Retrieval-Augmented Generation)**: en vez de mandarle al LLM todo
  el documento entero (caro y poco preciso), primero troceamos el texto en
  fragmentos pequeños ("chunks"), los convertimos en vectores numéricos
  ("embeddings") que capturan su significado, y solo le mandamos al LLM los
  fragmentos que de verdad son relevantes para la pregunta que le hacemos.
- **Embeddings locales**: el paso de "convertir texto en vectores" lo hace
  aquí un modelo pequeño que corre en tu propio ordenador (no en la nube),
  así que no cuesta dinero ni gasta tokens de ninguna API.

## Qué hace cada fichero

- `dd_copilot/ingest.py` — recibe una URL, la ruta a un PDF, la ruta a un
  fichero de texto, o texto pegado directamente, y lo convierte todo a
  texto plano.
- `dd_copilot/chunking.py` — trocea ese texto en fragmentos ("chunks") por
  frases, no por un número fijo de caracteres, para no cortar ideas a la
  mitad.
- `dd_copilot/index.py` — convierte cada fragmento en un embedding (con un
  modelo local, gratis) y construye un índice en memoria para poder
  buscar "qué fragmento es más relevante para esta pregunta" sin llamar al
  LLM.
- `dd_copilot/citation_check.py` — comprueba que una cita que dice haber
  sacado el LLM del texto, aparece de verdad ahí (con cierta tolerancia a
  variaciones menores). Si no aparece, se descarta — nunca se acepta un
  dato inventado.
- `dd_copilot/providers.py` — define cómo hablar con el LLM. Hay dos
  opciones: **Claude** (por defecto, recomendado, requiere una API key con
  saldo) u **Ollama** (un modelo que corre en tu ordenador, gratis, sin
  necesidad de API key, pero de calidad más limitada).
- `dd_copilot/extract.py` — hace las preguntas del checklist fijo
  (problema que resuelve, diferenciación técnica, rendimiento, y 4 riesgos)
  usando solo los fragmentos relevantes, y valida cada respuesta con
  `citation_check.py`.
- `dd_copilot/report.py` — junta todo en un informe Markdown con 5
  secciones fijas.
- `dd_copilot/pipeline.py` — encadena todos los pasos anteriores en una
  sola función.
- `dd_copilot/cli.py` — el comando de terminal `ddcopilot analyze`.
- `app.py` — una página web sencilla (Streamlit) para usar la herramienta
  sin terminal.

## Instalación paso a paso

1. Necesitas Python 3.12 o superior instalado. Comprueba tu versión:
   ```bash
   python3.12 --version
   ```
2. Crea un entorno virtual (un espacio aislado para las dependencias de
   este proyecto, para no mezclarlas con otros proyectos de tu ordenador):
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```
   Verás que el prompt de tu terminal cambia para indicar que el entorno
   está activo.
3. Instala el proyecto y sus dependencias:
   ```bash
   pip install -e ".[dev]"
   ```

## Configurar el acceso al LLM

Tienes dos opciones, no hace falta configurar las dos:

### Opción A — Claude (recomendado, requiere saldo en tu cuenta de Anthropic)

1. Copia el fichero de ejemplo: `cp .env.example .env`
2. Abre `.env` y pon tu clave real de la API de Anthropic:
   `ANTHROPIC_API_KEY=sk-ant-...`
   (la consigues en [console.anthropic.com](https://console.anthropic.com),
   sección "API Keys"; necesitas tener saldo en "Plans & Billing").
3. El fichero `.env` nunca se sube a GitHub (está en `.gitignore`).

### Opción B — Ollama en local (gratis, sin API key, calidad más limitada)

1. Instala [Ollama](https://ollama.com) si no lo tienes.
2. Descarga un modelo, por ejemplo:
   ```bash
   ollama pull llama3.1
   ```
3. No necesitas configurar nada más — al usar `--provider ollama`, la
   herramienta habla directamente con Ollama en tu ordenador.

## Cómo ejecutar el demo ya incluido

El demo con la startup real Isomorphic Labs ya está generado en
`examples/isomorphic-labs/report.md`. Para volver a generarlo tú mismo:

```bash
source .venv/bin/activate
ddcopilot analyze examples/isomorphic-labs/source.txt --output examples/isomorphic-labs/report.md --provider ollama
```

O, si tienes saldo en Claude, quitando `--provider ollama` usarás Claude
(el proveedor por defecto, de mayor calidad).

## Cómo analizar tu propia startup

```bash
# Con una URL:
ddcopilot analyze "https://una-startup-deeptech.com" --output mi-informe.md

# Con un PDF:
ddcopilot analyze ruta/al/whitepaper.pdf --output mi-informe.md

# Con texto pegado directamente:
ddcopilot analyze "Texto que has copiado de la web de la startup..." --output mi-informe.md
```

Añade `--provider ollama` a cualquiera de estos comandos si no tienes
saldo de Claude configurado.

O, si prefieres una interfaz visual en el navegador:

```bash
streamlit run app.py
```

## Cómo leer el informe generado

El informe (en inglés, ver la sección "Language decision" del README) tiene
5 secciones fijas:

1. **Executive Summary** — resumen de una frase de qué problema resuelve
   la startup.
2. **What the Startup Says** — lo que el material público sí afirma sobre
   el problema, la diferenciación técnica y el rendimiento, cada afirmación
   con su cita entre comillas. Si un campo dice **"Not mentioned in the
   source"**, significa que la herramienta no encontró ninguna afirmación
   verificable sobre ese punto — no que no exista, solo que no está en el
   material que le diste.
3. **What It Doesn't Say** — el checklist fijo de 4 riesgos técnicos
   (madurez tecnológica, dependencia de hardware/proveedor,
   reproducibilidad de resultados, riesgo regulatorio) que el material
   público NO cubre. Es información tan útil como lo que sí dice: te dice
   qué preguntar.
4. **Questions for the Next Founder Call** — una lista lista para usar en
   una llamada real con el equipo fundador, generada a partir de los
   huecos detectados en la sección anterior.
5. **Confidence Level** — un número del 1 al 5 con su justificación, que
   indica cuánta confianza tiene el propio análisis en sí mismo (no en la
   startup) — por ejemplo, baja si el material público era muy escaso.

## Cómo comprobar que todo sigue funcionando

```bash
pytest
```

Esto ejecuta toda la batería de pruebas (24 en total). Ninguna llamada real
a Claude ni a Ollama se hace durante los tests — todas las respuestas del
LLM están simuladas ("mockeadas"), así que ejecutar `pytest` no gasta
dinero ni requiere tener configurada ninguna clave.
