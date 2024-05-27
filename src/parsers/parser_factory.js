import { CSVParser } from './csv_parser'
export class ParserFactory {
    static createParser(fileType) {
        switch (fileType.toLowerCase()) {
            case 'text/csv':
                return new CSVParser();
            default:
                throw new Error(`Unsupported file type: ${fileType}`);
        }
    }
}
