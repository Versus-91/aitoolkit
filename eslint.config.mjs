import js from "@eslint/js";
import globals from "globals";

export default [
  js.configs.recommended,
  {
    rules: {
      "no-unused-vars": "warn",
    },
  },
  {
    languageOptions: {
      globals: {
        document: "readonly",
        console: "readonly",
        window: "readonly",
        tf: "readonly",
        Highcharts: "readonly",
        Plotly: "readonly",
        $: "readonly",
        setTimeout: "readonly",
        DataTable: "readonly",
        File: "readonly",




      }
    }
  }
];