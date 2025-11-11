"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

const featureData = [
  { feature: "Temperature", importance: 0.24, description: "Air temperature in Celsius" },
  { feature: "Humidity", importance: 0.21, description: "Relative humidity percentage" },
  { feature: "Wind Speed", importance: 0.18, description: "Wind speed in km/h" },
  { feature: "Drought Code", importance: 0.15, description: "Dryness index" },
  { feature: "Heat-Humidity Index", importance: 0.12, description: "Combined heat stress metric" },
  { feature: "Month (Cyclical)", importance: 0.1, description: "Seasonal encoding" },
]

const COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
  "hsl(var(--destructive))",
]

export function FeatureImportance() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Feature Importance Analysis</CardTitle>
          <CardDescription>Most influential factors in wildfire prediction</CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={{
              importance: { label: "Importance", color: "hsl(var(--chart-1))" },
            }}
            className="h-[400px]"
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureData} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 0.3]} />
                <YAxis dataKey="feature" type="category" />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {featureData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {featureData.map((feature, index) => (
          <Card key={feature.feature}>
            <CardHeader>
              <CardTitle className="text-base">{feature.feature}</CardTitle>
              <CardDescription>{feature.description}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold">{Math.round(feature.importance * 100)}%</span>
                <span className="text-sm text-muted-foreground">importance</span>
              </div>
              <div className="mt-3 h-2 rounded-full bg-secondary">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${feature.importance * 100}%`,
                    backgroundColor: COLORS[index % COLORS.length],
                  }}
                />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Understanding Feature Impact</CardTitle>
          <CardDescription>How these factors influence wildfire risk</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <h4 className="font-medium">Temperature</h4>
            <p className="text-sm text-muted-foreground">
              Higher temperatures increase evaporation rates, drying out vegetation and making it more flammable.
              Temperatures above 30°C significantly elevate fire risk.
            </p>
          </div>
          <div className="space-y-2">
            <h4 className="font-medium">Humidity</h4>
            <p className="text-sm text-muted-foreground">
              Low relative humidity accelerates fuel drying. When humidity drops below 30%, vegetation becomes
              critically dry and highly susceptible to ignition.
            </p>
          </div>
          <div className="space-y-2">
            <h4 className="font-medium">Wind Speed</h4>
            <p className="text-sm text-muted-foreground">
              Strong winds supply oxygen to fires, increase spread rates, and carry embers to new locations. Wind speeds
              above 25 km/h dramatically increase fire danger.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
