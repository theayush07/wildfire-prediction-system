"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

const modelComparison = [
  { model: "Logistic Regression", accuracy: 0.82, precision: 0.78, recall: 0.75, f1: 0.76 },
  { model: "Random Forest", accuracy: 0.91, precision: 0.89, recall: 0.88, f1: 0.88 },
  { model: "XGBoost", accuracy: 0.94, precision: 0.93, recall: 0.91, f1: 0.92 },
  { model: "LightGBM", accuracy: 0.93, precision: 0.92, recall: 0.9, f1: 0.91 },
]

const rocData = Array.from({ length: 11 }, (_, i) => ({
  fpr: i / 10,
  logistic: 0.5 + (i / 10) * 0.3,
  rf: 0.6 + (i / 10) * 0.35,
  xgboost: 0.7 + (i / 10) * 0.28,
  lightgbm: 0.68 + (i / 10) * 0.3,
}))

export function ModelPerformance() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Model Comparison</CardTitle>
          <CardDescription>Performance metrics across different ML algorithms</CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={{
              accuracy: { label: "Accuracy", color: "hsl(var(--chart-1))" },
              precision: { label: "Precision", color: "hsl(var(--chart-2))" },
              recall: { label: "Recall", color: "hsl(var(--chart-3))" },
              f1: { label: "F1 Score", color: "hsl(var(--chart-4))" },
            }}
            className="h-[400px]"
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelComparison}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" />
                <YAxis domain={[0, 1]} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Legend />
                <Bar dataKey="accuracy" fill="var(--color-accuracy)" name="Accuracy" />
                <Bar dataKey="precision" fill="var(--color-precision)" name="Precision" />
                <Bar dataKey="recall" fill="var(--color-recall)" name="Recall" />
                <Bar dataKey="f1" fill="var(--color-f1)" name="F1 Score" />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>ROC Curves</CardTitle>
            <CardDescription>True Positive Rate vs False Positive Rate</CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{
                logistic: { label: "Logistic Regression", color: "hsl(var(--chart-1))" },
                rf: { label: "Random Forest", color: "hsl(var(--chart-2))" },
                xgboost: { label: "XGBoost", color: "hsl(var(--chart-3))" },
                lightgbm: { label: "LightGBM", color: "hsl(var(--chart-4))" },
              }}
              className="h-[300px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rocData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="fpr" label={{ value: "False Positive Rate", position: "insideBottom", offset: -5 }} />
                  <YAxis label={{ value: "True Positive Rate", angle: -90, position: "insideLeft" }} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Legend />
                  <Line type="monotone" dataKey="logistic" stroke="var(--color-logistic)" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="rf" stroke="var(--color-rf)" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="xgboost" stroke="var(--color-xgboost)" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="lightgbm" stroke="var(--color-lightgbm)" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Best Model: XGBoost</CardTitle>
            <CardDescription>Detailed performance metrics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">ROC-AUC Score</p>
                <p className="text-2xl font-bold">0.94</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">PR-AUC Score</p>
                <p className="text-2xl font-bold">0.91</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Precision</p>
                <p className="text-2xl font-bold">93%</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Recall</p>
                <p className="text-2xl font-bold">91%</p>
              </div>
            </div>
            <div className="pt-4 border-t space-y-2">
              <p className="text-sm font-medium">Model Characteristics</p>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span>Handles imbalanced datasets effectively</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span>Fast training and prediction times</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">✓</span>
                  <span>Robust to overfitting with regularization</span>
                </li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
