# DD-Copilot — Diseño

## Contexto y objetivo

DD-Copilot es una herramienta que convierte documentación pública de una
startup deep-tech (web, whitepaper, PDF) en un informe de due diligence
técnica estructurado. Es el "Proyecto 0" del documento
`Prompts Proyectos IA - Deep Tech Analyst.md`, pensado como pieza de
portfolio para procesos de Tech Analyst en deep-tech (ref. MNTY).

Prioridades explícitas del usuario:
- Ahorro de tokens / eficiencia de coste en cada llamada al LLM.
- Trazabilidad no negociable: toda afirmación debe citar su fuente.
- Explicabilidad: el usuario es principiante en IA, necesita un documento
  final en español, sin jerga sin definir, que le permita entender y
  probar la herramienta él mismo.

## Ubicación y publicación

- Repo local: `~/Projects/dd-copilot` (convención de proyectos del usuario).
- Publicado en GitHub como repo público bajo la cuenta `jjpp01x`.

## Arquitectura

```
ingest.py        → normaliza URL/PDF/texto a texto plano
index.py         → chunking semántico + embeddings locales (LlamaIndex)
extract.py       → retrieval + extracción estructurada vía Claude
citation_check.py→ valida que cada cita exista literalmente en la fuente
report.py        → ensambla el informe Markdown final
cli.py           → comando `ddcopilot analyze <url|pdf>` (Typer + Rich)
app.py           → visor Streamlit de una sola página (reutiliza el core)
```

### 1. Ingesta (`ingest.py`)
- URL → `trafilatura` (extracción de contenido principal, sin boilerplate).
- PDF → `pypdf`.
- Texto pegado → se usa tal cual.
- Salida: texto plano normalizado + metadato de fuente (URL o nombre de fichero).

### 2. Indexado (`index.py`, LlamaIndex)
- `SentenceSplitter` de LlamaIndex para chunking semántico (por oraciones/
  párrafos, con solape, no por longitud fija de tokens).
- Embeddings locales vía `HuggingFaceEmbedding` (`sentence-transformers`,
  modelo `all-MiniLM-L6-v2`): coste cero de API en la fase de indexado.
- `VectorStoreIndex` en memoria (sin base vectorial externa — proyecto de
  tamaño demo, no requiere persistencia distribuida).

### 3. Extracción estructurada (`extract.py`)
- LLM: Claude vía `llama-index-llms-anthropic` (Anthropic SDK oficial).
- Checklist fijo a extraer:
  - Problema que resuelve la tecnología.
  - Diferenciación técnica frente a alternativas.
  - Afirmaciones de rendimiento/escalabilidad.
  - Riesgos técnicos NO mencionados (checklist fijo: madurez TRL,
    dependencia de hardware/proveedor, reproducibilidad de resultados,
    riesgo regulatorio si aplica).
- Salida forzada a JSON vía Pydantic (`structured_predict` de LlamaIndex),
  no texto libre — evita parsing frágil y reduce tokens de post-proceso.
- **Cascada de modelos para ahorro de tokens**:
  - Paso 1 (filtrado, sin LLM): similitud coseno de embeddings entre la
    pregunta de cada campo del checklist y los chunks — solo el top-k
    de chunks relevantes pasa a Claude.
  - Paso 2 (extracción por campo, barato): **Claude Haiku** clasifica y
    extrae de los chunks filtrados.
  - Paso 3 (síntesis final, calidad): **Claude Sonnet** recibe únicamente
    las extracciones ya estructuradas (no los chunks en bruto) y redacta
    el informe final en una única llamada.
  - Prompt caching de Anthropic para el system prompt fijo (reutilizado
    en todas las llamadas de un mismo análisis).

### 4. Validación de citas (`citation_check.py`)
- Requisito no negociable: cada afirmación del informe debe venir con
  una cita textual del fragmento fuente.
- Tras la extracción, se comprueba (fuzzy match, umbral configurable)
  que la cita existe literalmente en el chunk de origen.
- Si una afirmación no tiene cita verificable, se descarta y el campo
  correspondiente se marca explícitamente como "no mencionado en la
  fuente" en vez de inventarse contenido.

### 5. Informe final (`report.py`)
Plantilla Markdown con secciones fijas:
1. Resumen ejecutivo
2. Qué dice la startup (con citas)
3. Qué no dice (riesgos del checklist no cubiertos)
4. Preguntas para la siguiente llamada con el fundador
5. Nivel de confianza del análisis (1-5, justificado)

### 6. Interfaces
- **CLI** (`cli.py`, Typer + Rich): `ddcopilot analyze <url_o_ruta_pdf>`
  genera `informe.md` en el directorio de salida indicado.
- **Streamlit** (`app.py`): una sola página, input (URL/PDF/texto pegado)
  + botón "Analizar" + render del Markdown resultante. Reutiliza
  exactamente el mismo core que la CLI (sin lógica duplicada).

## Manejo de errores
- Fuente vacía o ilegible (PDF corrupto, URL caída): mensaje de error
  claro en CLI/Streamlit, sin llamar al LLM.
- Fallo de la API de Claude (rate limit, timeout): reintento con backoff
  exponencial (máx. 3 intentos), luego error explícito al usuario.
- Ninguna cita verificable para un campo: no se inventa — se marca como
  "no mencionado", nunca se omite en silencio.

## Testing (TDD)
- Unitarios:
  - Chunking: verifica que el `SentenceSplitter` produce chunks con
    solape esperado y sin cortar oraciones a la mitad.
  - `citation_check.py`: casos con cita exacta (pasa), cita alterada
    (falla), cita inexistente (falla).
  - `report.py`: ensamblado de plantilla con datos de ejemplo produce
    las 5 secciones fijas en el orden correcto.
- Integración:
  - Flujo completo `ingest → index → extract → report` con la llamada a
    Claude mockeada (fixture de respuesta JSON), para no gastar tokens
    reales en CI y mantener los tests deterministas.

## Demo
- Startup de ejemplo: **Isomorphic Labs** (spin-off de DeepMind, biotech +
  IA). Material público claro (web, comunicados), sin temas sensibles.
- El informe generado para este caso se incluye en el repo
  (`examples/isomorphic-labs/informe.md`) como demo reproducible.

## Entregables
1. Repo en `~/Projects/dd-copilot`, publicado en GitHub (`jjpp01x`, público).
2. `README.md` técnico: explica el "por qué" de cada decisión de
   arquitectura (chunking, cascada de modelos, validación de citas), no
   solo el "qué".
3. `GUIA-DE-USO.md`: documento en español, sin jerga sin definir, para
   que el usuario (principiante en IA) entienda cada pieza, instale el
   proyecto, ejecute el demo y sepa interpretar el informe generado.
4. Ejemplo ya procesado de Isomorphic Labs como demo.

## Fuera de alcance (roadmap, no implementar ahora)
- Comparación automática entre 2-3 startups del mismo vertical.
- Scoring cuantitativo ponderado entre startups.
- Soporte multi-proveedor de LLM (solo Claude por ahora).
