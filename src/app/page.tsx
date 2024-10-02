"use client"

import Link from 'next/link'
import { Button } from '@/components/button'
import { motion } from 'framer-motion'
import { ArrowRight, Recycle, Trash2, TreeDeciduous } from 'lucide-react'

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 relative overflow-hidden">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center"
      >
        <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">
          Welcome to EcoSort
        </h1>
        <p className="text-xl mb-8 max-w-2xl text-foreground">
          Upload images of waste items and get smart sorting and recycling recommendations.
          Help make our planet cleaner, one item at a time.
        </p>
        <Button asChild className="bg-green-500 hover:bg-green-600 text-white">
          <Link href="/upload" className="flex items-center">
            Start Sorting <ArrowRight className="ml-2" />
          </Link>
        </Button>
      </motion.div>
      <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
        {[
          { icon: Recycle, title: "Recycle", description: "Learn how to properly recycle various materials" },
          { icon: Trash2, title: "Reduce Waste", description: "Tips on reducing your overall waste production" },
          { icon: TreeDeciduous, title: "Go Green", description: "Discover more ways to live an eco-friendly lifestyle" },
        ].map((item, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 * (index + 1) }}
            className="bg-card text-card-foreground p-6 rounded-lg shadow-lg"
          >
            <item.icon className="h-12 w-12 text-green-500 mb-4" />
            <h2 className="text-xl font-semibold mb-2">{item.title}</h2>
            <p className="text-muted-foreground">{item.description}</p>
          </motion.div>
        ))}
      </div>
    </main>
  )
}