import { CSVParser } from './csv_parser'
import { XLXParser } from './xlx_parser'

export class ParserFactory {
    static createParser(fileType) {
        switch (fileType.toLowerCase()) {
            case 'text/csv':
                return new CSVParser();
            case 'xlsx':
                return new XLXParser();
            default:
                throw new Error(`Unsupported file type: ${fileType}`);
        }
    }
}
