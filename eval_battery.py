"""Evaluate routing/orchestration against the 18-question benchmark."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


CASES = [
    (
        1,
        "¿Cuál es el plazo máximo que tiene una Empresa de Servicios de Inversión (ESI) para comunicar a la CNMV la apertura de una nueva sucursal en España?",
        "legal",
    ),
    (
        2,
        "¿En qué fecha entra en vigor la Circular 2/2017 sobre la información de las instituciones de inversión colectiva (IIC) extranjeras?",
        "legal",
    ),
    (
        3,
        "Según el Acuerdo de 1998, ¿cómo se denomina el sistema de intercambio de información por línea telemática implementado por la CNMV?",
        "legal",
    ),
    (
        4,
        "¿En qué supuestos excepcionales NO es obligatorio que una ESI realice una comunicación previa a la CNMV al nombrar cargos de administración?",
        "legal",
    ),
    (
        5,
        "Si una entidad comercializadora de fondos no vende ninguna IIC extranjera durante todo un trimestre, ¿se libra de tener que presentar el estado estadístico a la CNMV?",
        "legal",
    ),
    (
        6,
        "Para que una copia impresa en papel de un documento recibido por CIFRADOC tenga plena validez ante terceros, ¿qué información obligatoria debe contener el certificado anexo firmado por el interesado?",
        "legal",
    ),
    (
        7,
        "Una SGIIC decide delegar la actividad de análisis y selección de inversiones de una IIC de Inversión Libre (IICIL). Además de avisar con 15 días hábiles, ¿qué comprobación extra y documento adicional exige la normativa enviar a la CNMV?",
        "legal",
    ),
    (
        8,
        "Según la Circular del Banco de España, en una emisión de bonos garantizados con estructuras de vencimiento prorrogable, ¿cuáles son los únicos cuatro eventos desencadenantes que justifican solicitar dicha prórroga del vencimiento?",
        "legal",
    ),
    (
        9,
        "¿Qué información exacta se exige a las entidades emisoras de bonos garantizados que remitan periódicamente sobre el órgano de control del conjunto de cobertura?",
        "legal",
    ),
    (
        10,
        "Si una ESI contrata a un nuevo agente, ¿cuándo debe comunicarlo a la CNMV y qué protocolo tecnológico exacto le obliga la normativa a utilizar?",
        "legal",
    ),
    (
        11,
        "Las normativas imponen calendarios distintos según el reporte. ¿Qué diferencia de plazo máximo existe entre la presentación trimestral del estado estadístico A01 de IIC extranjeras y la presentación de los estados del colchón de liquidez (LQB) para bonos garantizados?",
        "legal",
    ),
    (
        12,
        "Tanto las ESI como las entidades de crédito deben reportar cambios estructurales. Si se altera el grupo consolidable de una ESI, ¿qué plazo tiene para avisar a la CNMV, y si se modifica la organización del registro especial del conjunto de cobertura de una entidad de crédito, cuándo se remite al Banco de España?",
        "legal",
    ),
    (
        13,
        "En el estado BG_4-1 sobre bonos hipotecarios, ¿cuáles son los tres tramos porcentuales exactos en los que se debe desglosar obligatoriamente la distribución del loan to value de los inmuebles residenciales?",
        "technical",
    ),
    (
        14,
        "Al rellenar el estado LQB 3 de cálculo del colchón de liquidez, ¿qué código numérico concreto indica el manual que se introduzca en la celda del Porcentaje de Ratio de cobertura de liquidez si el valor de la salida neta acumulada máxima a cubrir es cero?",
        "technical",
    ),
    (
        15,
        "Para computar el colchón de liquidez en el estado LQB 1, ¿qué ponderación estándar (multiplicador) se debe aplicar a los Valores representativos de deuda de empresas que cuenten con un nivel de calidad crediticia 1?",
        "technical",
    ),
    (
        16,
        "En mi programa de bonos garantizados, todos los días de los próximos 180 días tendré más entradas de dinero que salidas. Legalmente, ¿qué horizonte temporal me obliga a cubrir la norma, y operativamente, qué número y signo debo reportar en la celda Salida neta de liquidez acumulada máxima del estado LQB 2?",
        "both",
    ),
    (
        17,
        "Tengo en mi cartera de cobertura primaria un préstamo hipotecario en impago. Legalmente, ¿a qué reglamento europeo debo acudir para la definición de impago, y técnicamente, bajo qué estado y sub-fila del reporte del Banco de España debo declarar este nominal?",
        "both",
    ),
    (
        18,
        "Hemos recibido un documento de la CNMV mediante el sistema telemático con claves. Jurídicamente, ¿qué dos principios garantiza que solo nosotros podamos leerlo y que no haya sido alterado? A nivel informático, ¿cómo interactúan la clave pública y privada del receptor en la recepción según el Anexo I?",
        "both",
    ),
]


def post(base_url: str, path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"{base_url}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", default="data/battery_eval_after.json")
    args = parser.parse_args()

    rows = []
    for num, question, expected in CASES:
        route_debug = post(args.base_url, "/route-debug", {"question": question})
        query_debug = post(
            args.base_url,
            "/query-debug",
            {
                "question": question,
                "top_k": args.top_k,
                "debug_probe_mode": "auto",
                "include_raw_text": False,
            },
        )
        rows.append(
            {
                "num": num,
                "expected": expected,
                "route_debug_final": route_debug.get("final_decision", {}).get("route"),
                "query_route_initial": query_debug.get("route_initial"),
                "query_route_final": query_debug.get("route_final"),
                "engines_used": query_debug.get("engines_used", []),
                "legal_ok": ((query_debug.get("legal_trace") or {}).get("evaluation") or {}).get("evidence_ok"),
                "technical_ok": ((query_debug.get("technical_trace") or {}).get("evaluation") or {}).get("evidence_ok"),
                "route_reason": query_debug.get("route_reason"),
            }
        )

    report = {
        "route_debug_acc": sum(1 for row in rows if row["route_debug_final"] == row["expected"]) / len(rows),
        "query_route_acc": sum(1 for row in rows if row["query_route_final"] == row["expected"]) / len(rows),
        "rows": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
