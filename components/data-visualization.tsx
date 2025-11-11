"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

const monthlyData = [
  { month: "Jan", fires: 12, avgTemp: 8, avgHumidity: 65 },
  { month: "Feb", fires: 15, avgTemp: 10, avgHumidity: 62 },
  { month: "Mar", fires: 28, avgTemp: 15, avgHumidity: 58 },
  { month: "Apr", fires: 42, avgTemp: 18, avgHumidity: 52 },
  { month: "May", fires: 55, avgTemp: 22, avgHumidity: 48 },
  { month: "Jun", fires: 78, avgTemp: 27, avgHumidity: 42 },
  { month: "Jul", fires: 95, avgTemp: 31, avgHumidity: 35 },
  { month: "Aug", fires: 105, avgTemp: 32, avgHumidity: 32 },
  { month: "Sep", fires: 88, avgTemp: 28, avgHumidity: 38 },
  { month: "Oct", fires: 52, avgTemp: 21, avgHumidity: 45 },
  { month: "Nov", fires: 25, avgTemp: 14, avgHumidity: 55 },
  { month: "Dec", fires: 18, avgTemp: 9, avgHumidity: 62 },
]

const distributionData = [
  { range: "0-10", count: 145 },
  { range: "10-25", count: 89 },
  { range: "25-50", count: 52 },
  { range: "50-100", count: 28 },
  { range: "100+", count: 15 },
]

export function DataVisualization() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Seasonal Fire Patterns</CardTitle>
          <CardDescription>Monthly fire occurrences with temperature and humidity trends</CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={{
              fires: { label: "Fire Incidents", color: "hsl(var(--chart-1))" },
              avgTemp: { label: "Avg Temperature (°C)", color: "hsl(var(--chart-2))" },
              avgHumidity: { label: "Avg Humidity (%)", color: "hsl(var(--chart-3))" },
            }}
            className="h-[400px]"
          >
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={monthlyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Area
                  type="monotone"
                  dataKey="fires"
                  stackId="1"
                  stroke="var(--color-fires)"
                  fill="var(--color-fires)"
                  fillOpacity={0.6}
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartContainer>
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Fire Size Distribution</CardTitle>
            <CardDescription>Number of fires by burned area (hectares)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {distributionData.map((item) => (
                <div key={item.range} className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">{item.range} ha</span>
                    <span className="text-muted-foreground">{item.count} fires</span>
                  </div>
                  <div className="h-2 rounded-full bg-secondary">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-orange-500 to-red-600"
                      style={{ width: `${(item.count / 145) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Dataset Summary</CardTitle>
            <CardDescription>UCI Forest Fires dataset statistics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Total Records</p>
                <p className="text-2xl font-bold">517</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Fire Events</p>
                <p className="text-2xl font-bold">329</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Features</p>
                <p className="text-2xl font-bold">12</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Location</p>
                <p className="text-2xl font-bold">Portugal</p>
              </div>
            </div>
            <div className="pt-4 border-t space-y-2">
              <p className="text-sm font-medium">Data Sources</p>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <span>UCI Machine Learning Repository</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <span>Montesinho Natural Park meteorological data</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <span>FWI (Fire Weather Index) system indices</span>
                </li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
