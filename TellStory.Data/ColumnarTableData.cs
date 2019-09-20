using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Text;

namespace TellStory.Data
{
    public class ColumnarTableData : StructuredData
    {
        private List<string> columnNames;
        private Dictionary<string, IList> columnData;
        private Dictionary<string, Type> columnTypes;
        private int size;
        private string tableName;

        public ColumnarTableData() : base()
        {
            columnNames = new List<string>();
            columnData = new Dictionary<string, IList>();
            columnTypes = new Dictionary<string, Type>();
            size = 0;
        }

        public ColumnarTableData(string name) : base(name)
        {
            columnNames = new List<string>();
            columnData = new Dictionary<string, IList>();
            columnTypes = new Dictionary<string, Type>();
            size = 0;
        }

        public ColumnarTableData(List<string> columnNames, Dictionary<string, IList> columnData, Dictionary<string, Type> columnTypes) : base()
        {
            this.columnNames = columnNames;
            this.columnData = columnData;
            this.columnTypes = columnTypes;
            size = columnData[columnNames[0]].Count;
        }

        public ColumnarTableData(string name, List<string> columnNames, Dictionary<string, IList> columnData, Dictionary<string, Type> columnTypes) : base(name)
        {
            this.columnNames = columnNames;
            this.columnData = columnData;
            this.columnTypes = columnTypes;
            size = columnData[columnNames[0]].Count;
        }

        public void AddColumn(string columnName, Type type)
        {
            columnNames.Add(columnName);

            if (type == typeof(Int32))
                columnData.Add(columnName, new List<int?>());
            else if (type == typeof(Int64))
                columnData.Add(columnName, new List<long?>());
            else if (type == typeof(float))
                columnData.Add(columnName, new List<float?>());
            else if (type == typeof(double))
                columnData.Add(columnName, new List<double?>());
            else if (type == typeof(DateTime))
                columnData.Add(columnName, new List<long?>());
            else if (type == typeof(string))
                columnData.Add(columnName, new List<String>());

            if (type == typeof(Int32))
                columnTypes.Add(columnName, typeof(Int32));
            else if (type == typeof(Int64))
                columnTypes.Add(columnName, typeof(Int64));
            else if (type == typeof(float))
                columnTypes.Add(columnName, typeof(float));
            else if (type == typeof(double))
                columnTypes.Add(columnName, typeof(double));
            else if (type == typeof(DateTime))
                columnTypes.Add(columnName, typeof(Int64));
            else if (type == typeof(string))
                columnTypes.Add(columnName, typeof(string));
        }

        public void AddRow(object[] values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                columnData[columnNames[i]].Add(values[i]);
            }
            size++;            
        }

        public void AddRow(string[] values)
        {
            for (int i=0; i<values.Length; i++)
            {
                Type type = columnTypes[columnNames[i]];                
                object convertedValue = null;

                if (type == typeof(Int32))
                    convertedValue = Int32.Parse(values[i]);
                else if (type == typeof(Int64))
                    convertedValue = Int64.Parse(values[i]);
                else if (type == typeof(float))
                    convertedValue = float.Parse(values[i]);
                else if (type == typeof(double))
                    convertedValue = double.Parse(values[i]);
                else if (type == typeof(DateTime))
                    //convertedValue = (long) (DateTime.Parse(values[i]).Subtract(new DateTime(1970, 1, 1)).TotalSeconds);
                    convertedValue = DateTime.Parse(values[i]);
                else if (type == typeof(string))
                    convertedValue = values[i];

                columnData[columnNames[i]].Add(convertedValue);                
            }
            size++;
        }

        public List<string> GetColumnNames()
        {
            return columnNames;
        }

        public Dictionary<string, Type> GetColumnTypes()
        {
            return columnTypes;
        }

        public Dictionary<string, IList> GetColumnData()
        {
            return columnData;
        }

        public static ColumnarTableData Create(TableData data)
        {
            ColumnarTableData columnTable = new ColumnarTableData(data.Name);

            foreach (DataColumn column in data.GetColumns())
            {
                string name = column.ColumnName;
                Type type = column.DataType;
                columnTable.GetColumnNames().Add(name);
                columnTable.GetColumnTypes().Add(name, column.DataType);

                if (type == typeof(Int32))
                    columnTable.GetColumnData().Add(name, new List<Int32>());
                else if (type == typeof(Int64))
                    columnTable.GetColumnData().Add(name, new List<Int64>());
                else if (type == typeof(float))
                    columnTable.GetColumnData().Add(name, new List<float>());
                else if (type == typeof(double))
                    columnTable.GetColumnData().Add(name, new List<double>());
                else if (type == typeof(DateTime))
                    columnTable.GetColumnData().Add(name, new List<DateTime>());
                else if (type == typeof(string))
                    columnTable.GetColumnData().Add(name, new List<string>());
            }

            for (int i = 0; i < data.GetCount(); i++)
            {                

                for (int j = 0; j < data.GetColumns().Count; j++)
                {
                    Type type = data.GetColumns()[j].DataType;

                    if (data.GetRow(i)[j] != DBNull.Value)
                    {
                        if (type == typeof(DateTime))
                        {
                            // Convert DateTime to unix timestamp
                            columnTable.GetColumnData()[data.GetColumns()[j].ColumnName].Add(((DateTime)data.GetRow(i)[j]).Subtract(new DateTime(1970, 1, 1)).TotalSeconds);
                        }
                        else
                        {
                            // Other types are just copied
                            columnTable.GetColumnData()[data.GetColumns()[j].ColumnName].Add(data.GetRow(i)[j]);
                        }
                    }
                    else
                    {
                        // Null values is inserted as null
                        columnTable.GetColumnData()[data.GetColumns()[j].ColumnName].Add(null);
                    }
                }
                columnTable.size++;
            }

            return columnTable;
        }

        public int GetSize()
        {
            return size;
        }

        public int GetCount()
        {
            return size;
        }

        public void Clear()
        {
            foreach (var columnName in columnNames)
            {
                columnData[columnName].Clear();
                size = 0;
            }
        }
    }
}
