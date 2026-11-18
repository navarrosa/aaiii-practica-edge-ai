// Página de tamizaje de retinopatía (S2, IN1042). Lee config.json: modelos, tamaño de entrada, etiquetas y ejemplos.
// Para usar tu propio modelo de imágenes solo cambias config.json (y copias tu .onnx a esta carpeta).
const $ = (id) => document.getElementById(id);
let CFG = null, sesion = null, infoSesion = null, ejemploActual = null;
const lienzo = $("lienzo"), ctx = lienzo.getContext("2d", { willReadFrequently: true });

(async () => {
  CFG = await leerConfig();
  document.title = CFG.titulo || document.title;
  lienzo.width = lienzo.height = CFG.entrada.lado;
  CFG.modelos.forEach((m, i) => $("selModelo").add(new Option(m.nombre, i)));
  (CFG.ejemplos || []).forEach((e, i) => {
    const b = document.createElement("button");
    b.className = "sec"; b.textContent = `Ejemplo ${i + 1}`;
    b.onclick = () => cargarImagen(e.archivo, e);
    $("ejemplos").appendChild(b);
  });
  $("dispositivo").textContent = describirDispositivo();
})();

$("btnCargar").onclick = async () => {
  const m = CFG.modelos[$("selModelo").value];
  $("estadoModelo").textContent = `Descargando ${m.archivo}…`;
  try {
    infoSesion = await cargarSesion(m.archivo, (t) => ($("estadoModelo").textContent = t));
    sesion = infoSesion.sesion;
    infoSesion.nombre = m.nombre;
    infoSesion.verificar = m.verificar;
    $("estadoModelo").textContent = `✔ ${m.nombre}: ${infoSesion.mb.toFixed(1)} MB · descarga ${(infoSesion.msDescarga / 1000).toFixed(1)} s · preparación ${(infoSesion.msCrear / 1000).toFixed(1)} s`;
    $("btnVelocidad").disabled = false;
    if (ejemploActual !== null || lienzo.dataset.lleno) await clasificar();
  } catch (e) {
    $("estadoModelo").textContent = "Error: " + e.message;
  }
};

$("archivo").onchange = (ev) => {
  const f = ev.target.files[0];
  if (f) cargarImagen(URL.createObjectURL(f), null);
};

function cargarImagen(url, ejemplo) {
  const img = new Image();
  img.onload = async () => {
    // Recorte cuadrado al centro y reducción al tamaño de entrada del modelo
    const s = Math.min(img.width, img.height);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, lienzo.width, lienzo.height);
    lienzo.dataset.lleno = "1";
    ejemploActual = ejemplo;
    await clasificar();
  };
  img.src = url;
}

// El modelo recibe los píxeles tal cual (0-255) en formato NHWC: (1, alto, ancho, 3).
// El reescalado a [-1, 1] va DENTRO del modelo, igual que en Keras. Si tu modelo espera otra cosa, cámbialo aquí.
function tensorDeEntrada() {
  const n = lienzo.width, rgba = ctx.getImageData(0, 0, n, n).data;
  const datos = new Float32Array(n * n * 3);
  const div = CFG.entrada.escala === "0-1" ? 255 : 1;
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    datos[j] = rgba[i] / div; datos[j + 1] = rgba[i + 1] / div; datos[j + 2] = rgba[i + 2] / div;
  }
  return new ort.Tensor("float32", datos, [1, n, n, 3]);
}

async function clasificar() {
  if (!sesion) { $("resultado").textContent = "Primero carga el modelo (paso 1)."; return; }
  const feeds = { [sesion.inputNames[0]]: tensorDeEntrada() };
  const t0 = performance.now();
  const salida = await sesion.run(feeds);
  const ms = performance.now() - t0;
  const p = salida[sesion.outputNames[0]].data[0];
  const positivo = p >= CFG.umbral;
  const r = $("resultado");
  r.className = "resultado " + (positivo ? "alerta" : "ok");
  r.textContent = `${positivo ? CFG.etiqueta_positiva : CFG.etiqueta_negativa} · P = ${p.toFixed(3)} · ${ms.toFixed(0)} ms`;
  let v = "";
  if (ejemploActual) {
    v = `Etiqueta real: ${ejemploActual.real ? CFG.etiqueta_positiva : CFG.etiqueta_negativa}.`;
    const ref = ejemploActual.p_python && ejemploActual.p_python[infoSesion.nombre];
    if (ref !== undefined && infoSesion.verificar === false) {
      // En INT8 los cálculos con enteros dependen del procesador (WebAssembly contra la CPU de la computadora)
      v += ` Python dio P = ${ref.toFixed(3)} en la computadora; en INT8 una diferencia es normal. El preprocesamiento se verifica con la versión float32.`;
    } else if (ref !== undefined) {
      v += ` Python dio P = ${ref.toFixed(3)} con este mismo modelo: ${Math.abs(ref - p) < 0.01 ? "✔ coinciden (el preprocesamiento es igual)" : "✖ revisa el preprocesamiento"}.`;
    }
  }
  $("verificacion").textContent = v;
}

$("btnVelocidad").onclick = async () => {
  if (!sesion) return;
  $("btnVelocidad").disabled = true;
  const feeds = { [sesion.inputNames[0]]: tensorDeEntrada() };
  const m = await medirLatencia(sesion, feeds, 20);
  const tabla = $("tablaTiempos");
  if (!tabla.rows.length) tabla.innerHTML = "<tr><th>Modelo</th><th>MB</th><th>ms (mediana)</th><th>mín-máx</th></tr>";
  tabla.insertRow().innerHTML = `<td>${infoSesion.nombre}</td><td>${infoSesion.mb.toFixed(1)}</td><td><b>${m.mediana.toFixed(0)}</b></td><td>${m.min.toFixed(0)}-${m.max.toFixed(0)}</td>`;
  $("btnVelocidad").disabled = false;
};
