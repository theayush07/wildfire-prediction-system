"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { PredictionForm } from "@/components/prediction-form"
import { ModelPerformance } from "@/components/model-performance"
import { FeatureImportance } from "@/components/feature-importance"
import { DataVisualization } from "@/components/data-visualization"
import { Flame, BarChart3, Lightbulb, Database } from "lucide-react"

export function WildfirePredictionApp() {
  const [predictionResult, setPredictionResult] = useState<{
    risk: number
    riskLevel: "Low" | "Medium" | "High" | "Critical"
    factors: string[]
  } | null>(null)

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-amber-50 to-red-50 dark:from-gray-950 dark:via-gray-900 dark:to-gray-950">
      {/* Header */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-orange-500 to-red-600">
              <Flame className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">Wildfire Prediction System</h1>
              <p className="text-xs text-muted-foreground">AI-Powered Risk Assessment</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container py-8 px-4">
        <Tabs defaultValue="predict" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 lg:w-auto">
            <TabsTrigger value="predict" className="gap-2">
              <Flame className="h-4 w-4" />
              <span className="hidden sm:inline">Predict</span>
            </TabsTrigger>
            <TabsTrigger value="performance" className="gap-2">
              <BarChart3 className="h-4 w-4" />
              <span className="hidden sm:inline">Performance</span>
            </TabsTrigger>
            <TabsTrigger value="features" className="gap-2">
              <Lightbulb className="h-4 w-4" />
              <span className="hidden sm:inline">Features</span>
            </TabsTrigger>
            <TabsTrigger value="data" className="gap-2">
              <Database className="h-4 w-4" />
              <span className="hidden sm:inline">Data</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="predict" className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <PredictionForm onPredictionComplete={setPredictionResult} />
              </div>
              <div>
                {predictionResult ? (
                  <Card>
                    <CardHeader>
                      <CardTitle>Risk Assessment</CardTitle>
                      <CardDescription>Based on current conditions</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="text-center space-y-2">
                        <div
                          className={`inline-flex h-24 w-24 items-center justify-center rounded-full text-3xl font-bold ${
                            predictionResult.riskLevel === "Critical"
                              ? "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-400"
                              : predictionResult.riskLevel === "High"
                                ? "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-400"
                                : predictionResult.riskLevel === "Medium"
                                  ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-950 dark:text-yellow-400"
                                  : "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-400"
                          }`}
                        >
                          {Math.round(predictionResult.risk * 100)}%
                        </div>
                        <div>
                          <p className="text-2xl font-bold">{predictionResult.riskLevel} Risk</p>
                          <p className="text-sm text-muted-foreground">Fire probability</p>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <p className="text-sm font-medium">Key Contributing Factors:</p>
                        <ul className="space-y-1 text-sm text-muted-foreground">
                          {predictionResult.factors.map((factor, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-orange-500">•</span>
                              <span>{factor}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </CardContent>
                  </Card>
                ) : (
                  <Card className="border-dashed">
                    <CardContent className="flex h-full min-h-[300px] items-center justify-center p-6">
                      <div className="text-center space-y-2">
                        <Flame className="mx-auto h-12 w-12 text-muted-foreground/50" />
                        <p className="text-sm text-muted-foreground">
                          Enter weather and environmental data to predict wildfire risk
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="performance">
            <ModelPerformance />
          </TabsContent>

          <TabsContent value="features">
            <FeatureImportance />
          </TabsContent>

          <TabsContent value="data">
            <DataVisualization />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
