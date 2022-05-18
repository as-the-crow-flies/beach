const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin')

const ROOT = path.resolve( __dirname, 'src' );
const DESTINATION = path.resolve( __dirname, 'dist' );

module.exports = {
    context: ROOT,

    entry: {
        'main': './main.ts'
    },
    
    output: {
        filename: '[name].bundle.js',
        path: DESTINATION
    },

    resolve: {
        extensions: ['.ts', '.js'],
        modules: [
            ROOT,
            'node_modules'
        ]
    },

    plugins: [
        new HtmlWebpackPlugin({
            template: './index.html'
        })
    ],

    module: {
        rules: [
            /****************
            * PRE-LOADERS
            *****************/
            {
                enforce: 'pre',
                test: /\.js$/,
                use: 'source-map-loader'
            },

            /****************
            * LOADERS
            *****************/
            {
                test: /\.ts$/,
                use: 'ts-loader'
            },
            {
                test: /\.png|.jpg$/,
                use: 'file-loader'
            },
            {
                test: /\.wgsl$/,
                use: 'raw-loader'
            }
        ]
    },

    devtool: 'cheap-module-source-map',
    devServer: {}
};

