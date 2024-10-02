import '@/styles/globals.css'
import { Inter } from 'next/font/google'
import { cn } from '@/lib/utils'
import { ThemeProvider } from '@/components/ui/theme-provider'
import { MainNav } from '@/components/ui/main-nav'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'EcoSort - Smart Garbage Collection',
  description: 'Upload images of waste for smart sorting and recycling recommendations.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={cn(
        "min-h-screen bg-[#F0F7F4] dark:bg-[#1A2B32] font-sans antialiased",
        inter.className
      )}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <div className="relative flex min-h-screen flex-col">
            <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))]"></div>
            <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
              <div className="container flex h-14 items-center">
                <MainNav />
              </div>
            </header>
            <div className="flex-1 relative z-10">{children}</div>
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}