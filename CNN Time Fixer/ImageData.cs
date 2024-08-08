﻿using Microsoft.ML.Data;

namespace CNN_Time_Fixer;

public class ImageData
{
    [LoadColumn(0)]
    public string ImagePath { get; set; }

    [LoadColumn(1)]
    public string Label { get; set; }
}