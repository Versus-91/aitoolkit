import { DataParser } from './parser';
import Papa from 'papaparse';

export class CSVParser extends DataParser {
    parse(content) {
        return new Promise((resolve, reject) => {
            Papa.parse(content, {
                worker: false,
                header: true,
                transform: (val) => {
                    if (val === "?" || val === "NA") {
                        return NaN
                    }
                    return val
                },
                // transformHeader: (val) => {
                //     return val.replace(/[^a-zA-Z0-9 ]/g, "").trim()
                // },
                skipEmptyLines: true,
                dynamicTyping: true,
                complete: async function (result) {
                    resolve(result.data)
                }
            })
        }
        )
    }
}